# DESCENT — resume state

Paused 2026-07-31 ~13:45 IST. Everything below is verified against the running
build, not remembered. Pick up at **"Next actions"**.

Standing instruction from the user: **quality at the highest possible end, cost
is not a constraint.** "Working is the floor, not the goal." Do not stop at
functional. `/loop` every subsystem against a critic until the critic runs out of
specific defects to name.

---

## How to run it

```bash
npm run dev                      # http://127.0.0.1:5173
node tools/capture/capture.mjs --poses          # 16 retina stills
node tools/capture/capture.mjs --seq launch --frames 48   # motion + contact sheet
node tools/capture/capture.mjs --all
```

`npx tsc --noEmit` → **0 errors**. The game boots with **zero page exceptions**
on ANGLE Metal (Apple M5 Pro) in headless Chromium.

Nothing is committed since `1a049b3` (foundation). ~29,000 lines are uncommitted
but safely on disk.

---

## What is DONE and verified

| Subsystem | Files | State |
|---|---|---|
| core | Engine, Input, MathX, Noise, RNG | complete |
| npr | Palette, ShaderChunks, CelMaterial, OutlineGeometry, GeneratedTextures, Sky, ShadowCascades, LUT, PostPipeline + 6 passes | complete, works on characters |
| terrain | Heightfield, Erosion, Zones, TerrainMaterial, TerrainClipmap, Scatter, Foliage, index | runs; **quality is the problem** |
| track | TrackSpline, TrackRibbon, Checkpoints, Furniture, index | working |
| bike | Wheel, BikeModel, BikeVisual, BikePhysics, TrickSystem, index | runs, **untuned** |
| rider | Skeleton, RiderMesh, Poses, IK, RiderRig, index | strongest subsystem |
| ai | AIRider, Personality, RaceDirector, PlayerRacer, Ghost, Replay | working |
| fx | CameraDirector, DustSystem, Debris, SpeedFX, ImpactFrames | built, **not emitting** |
| hud | HudCanvas, Hud, Widgets, Menus, Typeface | strong |
| audio | Synths, AudioEngine | built, **never heard** |
| game | Contracts (frozen), WorldConstants, Game | integrated |

### Verified measurements (not claims)
- Rider rig: max locked end-effector error **0.42 mm** over 300 frames of
  random-dt cruising (30–240 fps), all 10 TrickKinds, 4 crash directions.
- Landing absorption stagger: **67 ms legs → 133 ms spine → 183 ms head**.
- Terrain headless: 2.0 s generation, 21k rocks + 52k plants,
  **45–57 draw calls**, 120–235k tris/pass, clipmap 94k tris.

---

## KNOWN DEFECTS — the work queue, roughly in priority order

### P0 — terrain, drags down every wide shot (subsystem ~40%)
1. **Zone classifier misfires.** Snow covers the technical-start section that
   should be exposed rock. Dirt lands as scattered orange *splats* across the
   snowfield — reads as spilled paint, not a material boundary.
   → `src/terrain/Zones.ts`, the altitude/slope/erosion thresholds and `ZONE`
   in `WorldConstants.ts`.
2. **Trail carve staircases.** Heightfield is 2 m/texel over 4 km; the trail is
   ~5–11 m wide, so the carve has 3–5 texels and produces visible 1–2 m
   terraces. **Architectural, not a tuning value.** Options: higher-res
   heightfield near the route, or make the ribbon mesh authoritative and let
   the carve only prevent z-fighting.
3. **Terrain shows almost no cel banding.** The largest surface in every frame
   is near-flat colour. The 3–4 hard bands that ARE the art direction do not
   read on it. They read fine on riders and rocks.
   → `src/terrain/TerrainMaterial.ts`, `ZONE_RAMPS`.
4. **Quantised fog not reading at distance** — far ridges are muddy instead of
   stacking as flat paper layers. This is the brief's "what will make your wide
   shots look painted."
5. Axis-aligned rectangular seams in the snow (terrain agent flagged this
   itself; zone texture is NearestFilter R8, boundaries are texel-blocky).

### P0 — shadows
6. **Nothing casts a shadow onto the terrain.** No shadow under the rider at
   all. `ShadowCascades` runs; the terrain is likely not registered as a
   receiver, or its shadow material isn't sampling the cascades.
   → `src/npr/ShadowCascades.ts`, `src/terrain/TerrainMaterial.ts`,
   `src/npr/passes/RenderLists.ts`.

### P1 — FX not firing
7. **No dust at 50 km/h. No speed lines.** `Effects` is constructed with
   `autoEmit: true` and `setSubject(playerState)` is called, but nothing
   emits. Check `Effects.update` reads `state.front/rear.slideAmount` and that
   `POST_STATE.speedLineIntensity` is actually driven.

### P1 — bike physics never felt
8. Physics is written and stable but **has never been played by a human**, and
   the user's bar is *feel*, not simulation. Specifically unvalidated:
   the ~140 ms pump window, lean authority crossover at 9.5 m/s, landing
   quality → speed retention curve, crash tumble readability.
   → `src/bike/BikePhysics.ts`, `BODY_TUNE`.

### P2 — rider rig gaps (self-reported by the agent, all real)
9. Crash separation capped at ~30 cm because the rig root IS the bike. Brief
   asked for "a proper physical tumble, not a respawn fade."
10. **Feet DO release** for Superman, Tailwhip, No-footer and crashes via a
    `footLock` channel. This contradicts the user's "feet stay on the pedals,
    no detaching, ever." **Flagged to the user; awaiting their call.** Hands
    never release except in a crash.
11. No self-collision — a deep frontflip tuck lets a forearm pass through a
    thigh.
12. Face is minimal (no mouth/eyes behind the lens); deltoids read as armour
    balls with a hard sleeve seam; gloves are fists with no fingers wrapping
    the grip; shorts have no fold detail.
13. Tabletop counter-lean is authored, not derived from the bike's actual roll.
14. A bailed trick unwinds along exactly the path it wound (blend is
    `trick.phase` in both directions).

### P2 — terrain rendering gaps (self-reported, all real)
15. Inverted hulls don't dither-fade — `createHullMaterial` ignores the LOD
    fade hooks, so a near rock/tree's outline is solid while its fill stipples.
16. Prepass doesn't see the fade either; in the 175–205 m band a dissolving
    LOD0 tree still writes G-buffer normals.
17. `celShade` is redirected by a **preprocessor macro**
    (`#define celShade(x) terrainShade(x)`) because `CelOptions` exposes no
    shading hook. Compiles on ANGLE; a stricter driver would break the terrain.
    **Clean fix: add a `shadeOverride` option to `CelOptions`.**
18. Clipmap frustum culling is weak — only 2 of 28 quadrants typically cull,
    because the camera sits inside every outer-ring AABB. Fix: split outer
    rings into 16 blocks.
19. Rock variety is 3 shapes, conifers 2. Non-uniform instance scale is
    off-limits (`toWorld` assumes uniform scale).

### P3 — sky (logged at milestone 0, partially addressed)
20. Horizon dither noise in the sky gradient.
21. `helmet` ramp rim light reads too hot/pale (`rimStrength: 0.85`).

---

## NOT STARTED AT ALL

- **Both critic sub-agents.** The user explicitly asked for a *stills critic*
  (composition, line work, banding, palette, silhouette) and a separate
  *motion critic* (animation quality, weight, camera feel, whether speed reads
  as speed). Both must be harsh and specific, naming defects like "outline
  breaks on the rider's shoulder at three-quarter view." **They must report
  defects and must not approve work to be agreeable.**
- **The `/loop` cycles.** Every subsystem against its critic until the critic
  runs out of specific defects.
- **Performance pass.** Target is locked 60 fps at retina on an M5 Pro with no
  frame spikes on jumps or crashes. Never measured in a real session.
- **Audio verification.** Nothing has been listened to.
- **Replay / biggest-air cinematic.** Wired, never exercised.
- **Milestone 9 polish pass.**

---

## Bugs already found and fixed (do not re-fix)

- `ShaderChunks.ts`: backticks inside GLSL template-literal comments terminated
  the template string. **NEVER put a backtick in a GLSL comment in this repo.**
- Three.js GLSL3 emits **no** `pc_fragColor` and **no** `gl_FragColor` alias.
  Every fragment shader must declare its own output — use the `GLSL_FRAG_OUT`
  chunk and write to `fragColor`.
- `MathX.damp()` had a nonsensical `** (1/1)`.
- `tsconfig.json` needed `"types": ["vite/client"]`.
- `Sky.ts` missing `LinearFilter` import.
- `IK.ts` line ~152: `Math.cos(Math.PI - p.minBend)` where `minBend` is the
  minimum *interior* angle. Inverted the constraint — every limb was pinned
  straight and every end effector overshot by up to 27 cm.
- `Heightfield.ts` `buildCorridorField` used a hard 2 m `METRES_PER_SAMPLE`
  instead of deriving from `size`; at any resolution but 2048 it splatted the
  corridor thousands of texels off-route and every corridor-gated query
  returned "nowhere near the trail" (zero trail rocks/grass/flowers).
- `BikeVisual.placeChain` indexed `points[n]` on a closed loop where
  `cumulative` has `n+1` entries. Crashed every capture pose past
  `bike-detail`.

---

## Architecture notes worth not re-deriving

- `src/game/Contracts.ts` is **frozen**. Every subsystem is written against it
  and nothing else. That is what let eight agents build in parallel.
- The `ROUTE` in `WorldConstants.ts` is authoritative. Terrain biases toward a
  soft wide corridor **before** erosion; jump lips, the ravine, the narrowed
  ridge and the stream channel are carved back **after** erosion, because
  erosion would silt them up and "the jump eroded away" is not debuggable.
- Bike space: **+Y up, +Z forward, +X to the LEFT.** Origin at mid-wheelbase on
  the axle line, suspension topped out.
- Two cross-links the factories cannot make themselves, done in `Game.load()`:
  `bike.linkTrick(racer.trick)` (or a tailwhip never spins the frame) and
  `attachRigToBike(rig, bike)` (or hands IK to a `BIKE_GEOM` fallback bar).
- Physics is fixed 120 Hz with an accumulator clamped at 6 steps; rendering
  interpolates. Resolution is governed by a rolling *median* of 40 frames.
- The capture harness **takes the clock** (`capture.takeControl()`) and steps a
  fixed dt, so two builds are comparable. Poses are pinned to fractions along
  the course in `SITUATIONS` in `Game.ts`.

---

## Critic pass 1 — stills (run 2026-07-31, full 16-frame set)

Verdict: **none of the sixteen frames met the bar.** Full defect list is in the
session; the code-level findings that were verified and FIXED:

- **F1 `applyQuantizedFog` multiplied the discrete plateau strength by the
  continuous `t`** — restoring a gradient and defeating the entire "quantised
  atmospheric depth" feature. One multiply. Removed.
- **F2 Terrain shadow acne** rendered as hard corduroy because the two-step
  penumbra quantisation turns dithered acne into a deliberate-looking pattern.
  Root cause: constant world bias (0.045 m) is hopeless against a 21.5° sun.
  Replaced with **normal-offset bias** + new `uShadowTexelWorld` uniform.
  (The critic also claimed quantisation ran before the cascade blend — it did
  not. That part of the diagnosis was wrong.)
- **F3 Terrain read as a smooth gradient.** Two causes: squared half-Lambert
  compressed the terrain's real N·L window into one band, and the ambient
  bounce was tinted by a *continuous* `N.y`. Fixed with an exposure remap
  (`ndlRaw * 1.15 + 0.16`), shadow applied as a band-index step-down instead of
  a pre-ramp multiply, and `upness` quantised to 3 steps.
- **F4 Sawtooth trail edge.** The trail zone was stamped by thresholding the
  *height blend* weight, which falls off along-track between carve samples —
  so the material boundary dipped once per sample. Now stamped from lateral
  distance only, strictly inside the ribbon mesh.
- **Zone classification**: snow blanketed the 630 m technical start. Added
  slope-shed + sun-aspect scour to the snow rule (snow 8.3% → 3.3%, rock
  46% → 49.3%); exempted snow from the deposition-fan override (the "orange
  splats"); start plateau now marks Rock, not Dirt.
- **Capture harness**: contact sheets embedded 30 retina PNGs in one page and
  killed the renderer, taking the rest of the run with it. Now downscales one
  frame at a time in-browser; sheet failures are non-fatal.

### Still open from critic pass 1 (not yet fixed)
- Riders unidentifiable past ~40 px — no rim/saturation floor at distance.
- Cast shadows missing on several riders entirely; soft blobs where present.
- **Terrain has no silhouette outline** — ridgelines against sky are unlined
  (terrain carries no hull, and Sobel has no depth discontinuity there).
- Sky dome rectangular seam artefacts, top centre; sky gradient is continuous,
  not four plateaus.
- Palette drift: dirt reads hotter than authored; vignette multiplies toward
  black instead of tinting toward `GRADE.shadowTint`.
- Dead uniform stroke weight — no visible curvature/distance taper.
- Outline tears: bike-detail right shoulder; a stray hairline from the rider to
  the frame edge in switchback-lean (loose hull vertex).
- **HUD text collisions** — "DESCENT PROFILE" over "TECHNICAL START" in 5+
  frames, triple overlap in streambed. Shipping blocker.
- Composition: finish gate occluded by a rival; horizons on the centre line.
- **Capture poses are offset by ~one course section** — `crash` shows no crash,
  `tabletop-air` shows no jump, `streambed` shows no water. Half the review set
  is not testing what it claims. Fix the `t` values in `SITUATIONS` in Game.ts.
- Ribbon skirt spikes as triangular teeth on steep cross-slopes.
- Terrain interior Sobel lines are drawing the 2 m height-texel grid as
  staircases across the snow.

## Next actions, in order

1. **Terrain zone classification** — get rock on the technical start, kill the
   dirt splats, make the boundaries read as deliberate hard cel edges.
2. **Terrain cel banding** — the mountain must show 3–4 hard bands. This is the
   single highest-leverage fix in the project.
3. **Shadows onto the terrain.**
4. **Quantised fog** so far ridges stack as flat paper layers.
5. **Trail carve staircase** — decide ribbon-authoritative vs higher-res carve.
6. **Wire the FX** so dust and speed lines actually fire.
7. **Stand up both critic agents** and start the `/loop` cycles.
8. **Play the bike** and tune `BODY_TUNE` for feel.
9. Performance pass to locked 60 fps at retina.

Self-assessment at pause: **~60% of the bar.** The skeleton is complete and the
character work is genuinely good; the mountain it sits on is the problem.
