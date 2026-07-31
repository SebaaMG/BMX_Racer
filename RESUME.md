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

## Where it stands

Three critic passes run (2 stills, 1 motion... then a 3rd stills + 2nd motion).
Six specialist agents completed a repair round; seven more are running as of
this writing. `npx tsc --noEmit` clean between agent edits. The game boots and
plays end to end.

**Both critics still say: zero of 16 stills and 1 of 8 sequences meet the bar.**
(`launch` is the one sequence a critic passed outright.) Two stills are called
"close enough that the gap is nameable": `rider-closeup` (the RIDER and BIKE
meet the bar; the terrain behind them does not) and `crash` (composition good,
disqualified by a ghosted AI pack and missing ridge ink).

---

## The bugs that mattered most, and what they teach

Written down because every one of them was invisible to typechecking, invisible
to "does it run", and only fell out of MEASURING something.

1. **`Game.render` fed a dimensionless multiplier where a frame delta was
   expected.** `Effects.beginFrame` returned `timeScale`, not a scaled dt. The
   entire visual half of the game advanced ONE SECOND per rendered frame
   (`uFxTime` read 92 after 90 frames). Three agents found it independently.
   It explains most of critic pass 1: the rig converging in two frames then
   freezing, the landing stagger collapsing, and dust being emitted, pooled and
   killed before surviving a single draw.
2. **`applyTrackCarve` used `if (w > weight[k])` against a weight that
   saturates at 1.0**, with carve points 3.5 m apart. Five or six consecutive
   points tied at exactly 1.0; the strict `>` kept whichever was visited first;
   the loop runs uphill-to-downhill. Every texel took its height from up to 5 m
   back UP the course. Error -3.11..+5.00 m became -0.39..-0.01 m.
3. **`TrackSpline.carveTabletop` added the mound height twice at one index** —
   two loops sharing endpoint `iTake`. 4.2 m of rise inside a 0.5 m plan step =
   83 degrees, which arc-length resampling then smeared into a centreline that
   was MULTIVALUED IN XZ. No heightfield can represent that, so the physics saw
   a mesa where the mesh drew a ramp and the bike fell through the mountain.
4. **`TileableNoise` never tiled.** `fbm01` wrapped on grid size while no octave
   frequency divided it (base: freqs 5,10,20,40,80 against a 32 lattice). Every
   "tileable" texture in the project had a seam at u=1 and v=1.
5. **The sky seam was the view ray's DERIVATIVE.** `vDir` was
   `normalize(position)` interpolated across the dome's triangles. The direction
   was fine; its derivative is piecewise-constant and jumps at every triangle
   edge — and the GPU picks the cloud mip level from exactly that. A dome
   meridian is a plane through the camera, so it projects to a dead-straight
   line.
6. **`applyQuantizedFog` multiplied the discrete plateau by the continuous t**,
   restoring the gradient the quantisation existed to remove. AND the plateau
   boundaries sat at 560/1175/1855 m while every shot occupies 200-900 m, so
   nothing could ever cross one. Fixing only the first would have changed
   nothing — which is what the critic caught on the next pass.
7. **Net roll stiffness was ~zero.** The balance loop's `inertiaRoll * leanKp`
   (2368 N.m/rad) was cancelled almost exactly by the gravitational tipping
   moment `m*g*h` (2415). Roll was a free integrator; the first bump decided
   which way the bike went. 27 degrees of lean with the stick centred.
8. **Capture poses spawned the bike origin at the trail SURFACE**, but the
   origin sits on the axle line one wheel radius above whatever it stands on.
   Both wheels spawned 0.27 m underground, the suspension answered with ~14 kN,
   and every frame ever captured was shot from a bike that had been launched.
   No wheel ever touched the ground, so no tyre force ever solved and no dust
   ever emitted.
9. **Water declared `specPower: 160` at a 21.5 degree sun** where N.H tops out
   near 0.4. `pow(0.4, 160)` is zero to every float in the machine. The stream
   carried a full specular declaration that could never produce a lit pixel.
10. **The terrain palette drain.** `plate` saturated at 0.43 by 58 m, mixing
    26.6% pure cool SKY_BOUNCE into everything beyond; and `upness` quantised
    `N.y = 1` to exactly 1.0, so on flat ground the bounce was pure sky and the
    warm GROUND_BOUNCE could never contribute. Together they drained ~half the
    committed chroma and rotated the trail hue up to 50 degrees off gold, past
    red into magenta. The stills critic proved the LUT was NOT at fault by
    re-running it by hand, then hand-propagated the authored colour through
    these two lines and matched the measured screen pixel to within 3/255.

### Standing rules this project learned the hard way
- **NEVER put a backtick inside a GLSL template-literal comment.** It
  terminates the template and breaks the TS parse with a cryptic error hundreds
  of lines away. Cost three cycles. Now a build error: `npm run check:glsl`,
  wired into `npm run build`.
- Three.js GLSL3 emits NO `pc_fragColor` and NO `gl_FragColor` alias. Every
  fragment shader declares its own output via the `GLSL_FRAG_OUT` chunk.
- **Critics are excellent at FINDING defects and unreliable at ATTRIBUTING
  them.** Every agent that dug in disproved part of its own brief — the sky
  seam was not the shafts, the camera collapse was an opponent in the lens, the
  trail sawtooth was ink not geometry. Always re-derive the cause.
- **Agents measure in isolation and can be wrong about the shipped build.** The
  camera agent measured a 254 px subject; the motion critic measured 110-150 px
  in the actual captures. Verify through the real path.

---

## Critic pass 3 (stills) + pass 2 (motion) — open defects

Ranked. Seven agents are working these now; anything still open when they
report goes back on this list.

- Terrain: the LARGEST surfaces still carry no bands (a column scan of the
  `treeline-silhouette` dune found 90 distinct values over 420 rows and zero
  hard steps).
- Ridgeline contour ink is an intermittent 1-px hairline that DROPS OUT
  mid-ridge (dips to lum 112 on one segment, 0 on the adjacent one; INK is 25).
- Nested concentric arcs / comb ripples over uncreased flat ground — the
  normal-Sobel has no scale invariance, so at raking angles it fires on a plane.
- A dead-straight full-width horizontal band across the horizon that composites
  OVER the clouds and slices them.
- God-ray shafts are now the largest smooth gradient left in the game.
- Motion smear still dissolves the rider in 5 of 16 frames, and reads as a
  bloom halo rather than a drawn streak.
- `bike-detail`: the camera is inside the dust volume; the bike renders as
  X-ray line art through the puffs.
- `ravine-gap`: no subject in frame, camera clipped into terrain.
- `landing` sequence: subject absent for the first 583 ms.
- AI riders past ~40 px are unidentifiable dark blobs — no chroma or rim floor.
- Water still has no surface, no flow contour, no highlight.
- HUD: the `7` tick descender cuts through `FINISH`; the speed numeral
  overflows its panel; the gate callout is near-invisible in fast frames.
- The rider has no face at close crop.
- Helmet specular is a soft bloom — the one thing reading as PBR on the rider.
- Trick 360 does not visibly rotate; impact frames still desaturate the frame.

## Not started
- Performance pass. Target is locked 60 fps at retina on an M5 Pro with no
  spikes on jumps or crashes. Never measured in a real session.
- Audio has never been listened to.
- Replay / biggest-air cinematic wired but never exercised.
