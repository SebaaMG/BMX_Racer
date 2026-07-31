/**
 * Hud — implements IHud. Layout, state, animation, and the overlay pass.
 *
 * ── HOW IT RENDERS ──────────────────────────────────────────────────────────
 * The HUD is a flat group of screen-space quads with a vertex program that
 * ignores the camera entirely, so it can be rendered by itself, last, straight
 * to the default framebuffer:
 *
 *     post.render(scene, camera, dt, time);
 *     hud.render(renderer);
 *
 * It is deliberately NOT added to the main scene. If it were, it would go
 * through the bloom threshold and the LUT grade, and a HUD that gets graded is a
 * HUD whose colours are no longer the colours you authored — the gold would
 * bloom, the ink would lift toward violet, and the whole thing would stop
 * reading as ink on top of the picture. `object` is still exposed because the
 * contract requires it, and it will render correctly if it is parented into a
 * scene, but the intended wiring is the explicit `render()` call.
 *
 * ── WHAT IT COSTS ───────────────────────────────────────────────────────────
 * Nothing is redrawn unless its content changed. In a steady racing frame that
 * is the clock (0.10 Mpx), the speed block (0.40 Mpx at ui=1) and, when the
 * marker has moved half a pixel, the route profile (0.19 Mpx). Everything else
 * — the standings board, the boost meter, the placement block, the menus — is a
 * cached texture and a single draw call. Live figures are on `hud.stats`.
 */

import {
  Group,
  Object3D,
  OrthographicCamera,
  WebGLRenderer,
} from 'three';
import type { HudModel, IHud } from '../game/Contracts';
import { RacePhase } from '../game/Contracts';
import { HUD_PALETTE } from '../npr/Palette';
import { clamp01, dampHL } from '../core/MathX';
import { HudCanvasRoot, HudLayer, type LayerPlacement } from './HudCanvas';
import { buildTypeface } from './Typeface';
import {
  BoostWidget,
  ClockWidget,
  CornerWidget,
  PlaceWidget,
  PopupWidget,
  RouteProfileWidget,
  SpeedWidget,
  StandingsWidget,
  WarningWidget,
  Widget,
} from './Widgets';
import { CountdownWidget, MenuScreen, type MenuKind, type ReplayFrameRect } from './Menus';

function place(
  anchor: LayerPlacement['anchor'],
  dx: number,
  dy: number,
  w: number,
  h: number,
): LayerPlacement {
  return { anchor, dx, dy, w, h };
}

export interface HudOptions {
  /** Start with the race furniture hidden (the game boots into the title). */
  initialPhase?: RacePhase;
}

export class Hud implements IHud {
  readonly object: Object3D;

  private root: HudCanvasRoot;
  private camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private byLayer = new Map<HudLayer, Widget>();
  private widgets: Widget[] = [];

  readonly profile: RouteProfileWidget;
  readonly clock: ClockWidget;
  readonly standings: StandingsWidget;
  readonly corner: CornerWidget;
  readonly place: PlaceWidget;
  readonly speed: SpeedWidget;
  readonly boost: BoostWidget;
  readonly popups: PopupWidget;
  readonly warning: WarningWidget;
  readonly countdown: CountdownWidget;
  readonly menu: MenuScreen;

  /** Live cost of the HUD, refreshed every frame. */
  readonly stats = { redrawMs: 0, layersRedrawn: 0, megapixels: 0, drawCalls: 0 };

  private scrimTarget = 0;
  private disposed = false;

  constructor(width = 1920, height = 1080, _opts: HudOptions = {}) {
    buildTypeface();

    this.root = new HudCanvasRoot(HUD_PALETTE.shadow);
    this.object = this.root.group;

    // ── Layer table. Insertion order is z-order. ────────────────────────────
    // ── THE PROFILE PANEL'S SIZE IS A COMPOSITION DECISION ────────────────────
    // It was 880 x 210 at (30, 24) — 46% of the design width and the whole
    // top-left quadrant. In eight of sixteen review frames it sat on the
    // horizon, which is the one line in a downhill racing shot that has to stay
    // readable. HUD furniture frames a picture; it does not stand in front of
    // it. 600 x 164 keeps every element the panel had (silhouette, skyline,
    // checkpoint hairlines, rival markers, the flooded ridden section) at
    // exactly the same internal proportions, occupies 31% of the width, and
    // clears the horizon in every pose in the set. It also ends at design x 608,
    // which puts 117 units of clean air between it and the clock's TIME label —
    // the second half of the header-collision fix.
    // 572 x 158 at (24, 16). The 600 x 164 that replaced the original
    // 880 x 210 was already clear of the horizon in all sixteen review poses —
    // verified, not assumed: the panel's bottom edge lands at design y 182 and
    // the highest horizon in the set is at y 300. So this pass is a trim rather
    // than a rescue: 8% off the area, 30 design units pulled back from the
    // right, and the slab is now sized so its SHEARED top-right corner lands
    // inside the backing store instead of 15 units outside it, which is what
    // was flattening that corner in every frame.
    this.profile = this.mount(new RouteProfileWidget(
      new HudLayer('profile', place('top-left', 24, 16, 572, 158)),
    ));
    this.clock = this.mount(new ClockWidget(
      new HudLayer('clock', place('top', 0, 22, 470, 200)),
    ));
    this.standings = this.mount(new StandingsWidget(
      new HudLayer('board', place('top-right', 26, 24, 430, 280)),
    ));
    this.corner = this.mount(new CornerWidget(
      new HudLayer('corner', place('top', 0, 232, 300, 190)),
    ));
    this.place = this.mount(new PlaceWidget(
      new HudLayer('place', place('bottom-left', 26, 22, 480, 260)),
    ));
    // 580 x 360. The dial band's polygon needs 1.089 * R of width and of height
    // around its centre; at 560 x 340 it did not have it, and the band was
    // delivered with its top and right razored off by the backing store. The
    // extra 20 x 20 is what makes the gauge a whole object.
    this.speed = this.mount(new SpeedWidget(
      new HudLayer('speed', place('bottom-right', 20, 16, 580, 360)),
    ));
    this.boost = this.mount(new BoostWidget(
      new HudLayer('boost', place('bottom', 0, 26, 680, 120)),
    ));
    this.popups = this.mount(new PopupWidget(
      new HudLayer('popups', place('right', 24, -80, 620, 520)),
    ));
    this.warning = this.mount(new WarningWidget(
      new HudLayer('warn', place('center', 0, -200, 1040, 200)),
    ));
    this.countdown = this.mount(new CountdownWidget(
      new HudLayer('countdown', place('center', 0, 0, 760, 560), { background: false }),
    ));
    // The menu canvas is the largest surface in the HUD, so its backing store is
    // capped: at ui = 2.4 an uncapped 1360×860 layer would be a 47 MB texture,
    // and menu type is big enough that 1.4× is indistinguishable.
    this.menu = this.mount(new MenuScreen(
      new HudLayer('menu', place('center', 0, 0, 1360, 860), { maxScale: 1.4 }),
    ));

    this.resize(width, height);
  }

  private mount<T extends Widget>(w: T): T {
    this.root.add(w.layer);
    this.byLayer.set(w.layer, w);
    this.widgets.push(w);
    return w;
  }

  // ── IHud ──────────────────────────────────────────────────────────────────

  resize(width: number, height: number): void {
    this.root.resize(width, height);
    // A resize reallocates backing stores, which invalidates the baked
    // furniture; layout() already flags that, but the widgets also need their
    // signatures cleared or a layer whose content did not change would be
    // uploaded blank.
    for (const w of this.widgets) w.layer.markFurniture();
  }

  update(model: HudModel, dt: number, time: number): void {
    if (this.disposed) return;

    for (const w of this.widgets) w.update(model, dt, time);

    // The scrim only exists to hold the menus off the race behind them. It is a
    // flat fill, never a blur — a blurred pause background would be the only
    // out-of-focus pixel in the entire game.
    const menuUp = model.phase === RacePhase.Paused || model.phase === RacePhase.Results;
    const titleUp = model.phase === RacePhase.Attract;
    this.scrimTarget = menuUp ? 0.68 : titleUp ? 0.42 : 0;
    this.root.scrim.alpha = dampHL(this.root.scrim.alpha, this.scrimTarget, 0.09, dt);
    if (Math.abs(this.root.scrim.alpha - this.scrimTarget) < 0.003) this.root.scrim.alpha = this.scrimTarget;

    // Bake any furniture that a resize or a menu change invalidated, then redraw
    // only the layers whose signature moved.
    for (const w of this.widgets) {
      if (w.layer.furnitureDirty) {
        w.layer.drawFurniture((ctx, cw, ch) => w.furniture(ctx, cw, ch));
        w.layer.dirty = true;
      }
    }

    this.root.flush((layer, ctx) => {
      const w = this.byLayer.get(layer);
      if (w) w.draw(ctx, layer.w, layer.h);
    });

    this.stats.redrawMs = this.root.stats.redrawMs;
    this.stats.layersRedrawn = this.root.stats.layersRedrawn;
    this.stats.megapixels = this.root.stats.megapixels;
    let calls = this.root.scrim.mesh.visible ? 1 : 0;
    for (const l of this.root.layers) if (l.mesh.visible) calls++;
    this.stats.drawCalls = calls;
  }

  /**
   * Draw the overlay. Call once per frame, AFTER the post pipeline has written
   * the final image to the default framebuffer.
   */
  render(renderer: WebGLRenderer): void {
    if (this.disposed) return;
    const prev = renderer.getRenderTarget();
    const prevAuto = renderer.autoClear;
    renderer.autoClear = false;
    renderer.setRenderTarget(null);
    renderer.render(this.object, this.camera);
    renderer.autoClear = prevAuto;
    if (prev) renderer.setRenderTarget(prev);
  }

  dispose(): void {
    this.disposed = true;
    this.root.dispose();
    this.byLayer.clear();
    this.widgets.length = 0;
  }

  // ── Menu control ──────────────────────────────────────────────────────────
  //
  // The HUD does not read input — it has no idea what a key is. The game drives
  // the cursor through these and reads `menuSelection` back when it wants to
  // act on a confirm.

  get menuKind(): MenuKind {
    return this.menu.kind;
  }

  get menuItems(): string[] {
    return this.menu.items;
  }

  get menuSelection(): number {
    return this.menu.selection;
  }

  setMenuItems(items: string[]): void {
    this.menu.setItems(items);
  }

  setMenuSelection(i: number): void {
    this.menu.setSelection(i);
  }

  moveMenuSelection(delta: number): void {
    this.menu.moveSelection(delta);
  }

  /**
   * The results screen's replay window, in 0..1 screen UV. Hand this to the
   * camera director so the biggest-air replay is framed inside the hole the HUD
   * cut for it rather than behind the results table.
   */
  get replayFrame(): ReplayFrameRect {
    return this.menu.replayFrame;
  }

  /** Wipe per-run state (splits already shown, live popups). Call on restart. */
  resetRun(): void {
    this.clock.reset();
    this.popups.clear();
  }
}
