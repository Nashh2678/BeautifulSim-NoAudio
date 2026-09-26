# Sounds (not included)

This repo ships **no sound files**, only `manifest.json`, which links every game event to file names.
Drop your own audio files into this folder with those names and the visualizer picks them up at the next
start: `.ogg` or `.wav`, 48 kHz stereo (`sample_rate` in the manifest).

- Any subset works: an event with no file just stays silent, and when an event lists several files one is
  picked at random each time (variation).
- Rename or add entries in `manifest.json` to use your own file names or more variants.
- The spectated car's **engine** is synthesised from 10 short loops in `engine_src/` (`engine_src/1072222933.wav`, `engine_src/971492431.wav`, `engine_src/924662072.wav`, `engine_src/405608813.wav`, `engine_src/95566843.wav`, `engine_src/632240338.wav`, `engine_src/943180388.wav`, `engine_src/790896015.wav`, `engine_src/286033530.wav`),
  blended by engine RPM, throttle and load (`src/engine_synth.py`). Without all of them the engine is silent.
- Clips (C) include the sounds that are installed; with none, clips are video-only.
- Sound files in this folder are ignored by git (`.gitignore`), so they never get uploaded by accident.

| event | when | files (any subset) |
|---|---|---|
| `ball_bounce_floor_close` | ball bounces on the floor (close) | `ball_bounce_floor_close_0.ogg`, `ball_bounce_floor_close_1.ogg`, `ball_bounce_floor_close_2.ogg`, `ball_bounce_floor_close_3.ogg`, `ball_bounce_floor_close_4.ogg`, `ball_bounce_floor_close_5.ogg`, `ball_bounce_floor_close_6.ogg`, `ball_bounce_floor_close_7.ogg` |
| `ball_bounce_floor_far` | ball bounces on the floor (far) | `ball_bounce_floor_far_0.ogg`, `ball_bounce_floor_far_1.ogg`, `ball_bounce_floor_far_2.ogg`, `ball_bounce_floor_far_3.ogg`, `ball_bounce_floor_far_4.ogg`, `ball_bounce_floor_far_5.ogg`, `ball_bounce_floor_far_6.ogg`, `ball_bounce_floor_far_7.ogg` |
| `ball_bounce_floor_mid` | ball bounces on the floor (mid) | `ball_bounce_floor_mid_0.ogg`, `ball_bounce_floor_mid_1.ogg`, `ball_bounce_floor_mid_2.ogg`, `ball_bounce_floor_mid_3.ogg`, `ball_bounce_floor_mid_4.ogg`, `ball_bounce_floor_mid_5.ogg`, `ball_bounce_floor_mid_6.ogg`, `ball_bounce_floor_mid_7.ogg` |
| `ball_bounce_wall_close` | ball bounces on a wall (close) | `ball_bounce_wall_close_0.ogg`, `ball_bounce_wall_close_1.ogg`, `ball_bounce_wall_close_2.ogg`, `ball_bounce_wall_close_3.ogg`, `ball_bounce_wall_close_4.ogg`, `ball_bounce_wall_close_5.ogg`, `ball_bounce_wall_close_6.ogg`, `ball_bounce_wall_close_7.ogg` |
| `ball_bounce_wall_far` | ball bounces on a wall (far) | `ball_bounce_wall_far_0.ogg`, `ball_bounce_wall_far_1.ogg`, `ball_bounce_wall_far_2.ogg`, `ball_bounce_wall_far_3.ogg`, `ball_bounce_wall_far_4.ogg`, `ball_bounce_wall_far_5.ogg`, `ball_bounce_wall_far_6.ogg`, `ball_bounce_wall_far_7.ogg` |
| `ball_bounce_wall_mid` | ball bounces on a wall (mid) | `ball_bounce_wall_mid_0.ogg`, `ball_bounce_wall_mid_1.ogg`, `ball_bounce_wall_mid_2.ogg`, `ball_bounce_wall_mid_3.ogg`, `ball_bounce_wall_mid_4.ogg`, `ball_bounce_wall_mid_5.ogg`, `ball_bounce_wall_mid_6.ogg`, `ball_bounce_wall_mid_7.ogg` |
| `ball_hit_close` | car hits the ball (close) | `ball_hit_close_0.ogg`, `ball_hit_close_1.ogg`, `ball_hit_close_2.ogg`, `ball_hit_close_3.ogg`, `ball_hit_close_4.ogg`, `ball_hit_close_5.ogg`, `ball_hit_close_6.ogg`, `ball_hit_close_7.ogg` |
| `ball_hit_far` | car hits the ball (far) | `ball_hit_far_0.ogg`, `ball_hit_far_1.ogg`, `ball_hit_far_2.ogg`, `ball_hit_far_3.ogg`, `ball_hit_far_4.ogg`, `ball_hit_far_5.ogg`, `ball_hit_far_6.ogg`, `ball_hit_far_7.ogg` |
| `ball_hit_mid` | car hits the ball (mid) | `ball_hit_mid_0.ogg`, `ball_hit_mid_1.ogg`, `ball_hit_mid_2.ogg`, `ball_hit_mid_3.ogg`, `ball_hit_mid_4.ogg`, `ball_hit_mid_5.ogg`, `ball_hit_mid_6.ogg`, `ball_hit_mid_7.ogg` |
| `ball_post_close` | ball hits a goal post / crossbar (close) | `ball_post_close_0.ogg`, `ball_post_close_1.ogg`, `ball_post_close_2.ogg`, `ball_post_close_3.ogg`, `ball_post_close_4.ogg`, `ball_post_close_5.ogg`, `ball_post_close_6.ogg`, `ball_post_close_7.ogg` |
| `ball_post_far` | ball hits a goal post / crossbar (far) | `ball_post_far_0.ogg`, `ball_post_far_1.ogg`, `ball_post_far_2.ogg`, `ball_post_far_3.ogg`, `ball_post_far_4.ogg`, `ball_post_far_5.ogg`, `ball_post_far_6.ogg`, `ball_post_far_7.ogg` |
| `ball_post_mid` | ball hits a goal post / crossbar (mid) | `ball_post_mid_0.ogg`, `ball_post_mid_1.ogg`, `ball_post_mid_2.ogg`, `ball_post_mid_3.ogg`, `ball_post_mid_4.ogg`, `ball_post_mid_5.ogg`, `ball_post_mid_6.ogg`, `ball_post_mid_7.ogg` |
| `body_local` | the car you're watching hits the floor / a wall / the ceiling with its roof, side or nose | `body_local_0.ogg`, `body_local_1.ogg`, `body_local_2.ogg`, `body_local_3.ogg`, `body_local_4.ogg`, `body_local_5.ogg`, `body_local_6.ogg`, `body_local_7.ogg` |
| `body_other` | another car hits the floor / a wall / the ceiling with its body (3D) | `body_other_0.ogg`, `body_other_1.ogg`, `body_other_2.ogg`, `body_other_3.ogg`, `body_other_4.ogg`, `body_other_5.ogg`, `body_other_6.ogg`, `body_other_7.ogg` |
| `boost_loop` | boost held (looped, pitched with speed) | `boost_loop_alpha.wav` |
| `boost_start` | boost pressed | `boost_start_alpha.wav` |
| `boost_stop` | boost released | `boost_stop_alpha.wav` |
| `bump_0` | car bump (0 = light .. 3 = hard) | `bump_0_0.ogg`, `bump_0_1.ogg`, `bump_0_2.ogg`, `bump_0_3.ogg`, `bump_0_4.ogg`, `bump_0_5.ogg`, `bump_0_6.ogg`, `bump_0_7.ogg` |
| `bump_1` | car bump (0 = light .. 3 = hard) | `bump_1_0.ogg`, `bump_1_1.ogg`, `bump_1_2.ogg`, `bump_1_3.ogg`, `bump_1_4.ogg`, `bump_1_5.ogg`, `bump_1_6.ogg`, `bump_1_7.ogg` |
| `bump_2` | car bump (0 = light .. 3 = hard) | `bump_2_0.ogg`, `bump_2_1.ogg`, `bump_2_2.ogg`, `bump_2_3.ogg`, `bump_2_4.ogg`, `bump_2_5.ogg`, `bump_2_6.ogg`, `bump_2_7.ogg` |
| `bump_3` | car bump (0 = light .. 3 = hard) | `bump_3_0.ogg`, `bump_3_1.ogg`, `bump_3_2.ogg`, `bump_3_3.ogg`, `bump_3_4.ogg`, `bump_3_5.ogg`, `bump_3_6.ogg`, `bump_3_7.ogg` |
| `demo` | any demolition (main explosion) | `demo_0.ogg`, `demo_1.ogg`, `demo_2.ogg`, `demo_3.ogg`, `demo_4.ogg` |
| `demo_small_local` | the car you're watching gets demolished (its own layer) | `demo_small_local_0.ogg`, `demo_small_local_1.ogg`, `demo_small_local_2.ogg`, `demo_small_local_3.ogg`, `demo_small_local_4.ogg` |
| `demo_small_other` | another car gets demolished (its own layer, 3D) | `demo_small_other_0.ogg`, `demo_small_other_1.ogg`, `demo_small_other_2.ogg`, `demo_small_other_3.ogg`, `demo_small_other_4.ogg` |
| `demolish_stinger` | the car you're watching demolishes someone (the Demolition jingle) | `demolish_stinger_0.ogg` |
| `dodge_local` | flip / dodge (the car you're watching) | `dodge_local_0.ogg`, `dodge_local_1.ogg`, `dodge_local_2.ogg`, `dodge_local_3.ogg` |
| `dodge_other` | flip / dodge (other cars, 3D) | `dodge_other_0.ogg`, `dodge_other_1.ogg`, `dodge_other_2.ogg`, `dodge_other_3.ogg` |
| `doublejump_local` | double jump (the car you're watching) | `doublejump_local_0.ogg`, `doublejump_local_1.ogg`, `doublejump_local_2.ogg`, `doublejump_local_3.ogg` |
| `doublejump_other` | double jump (other cars, 3D) | `doublejump_other_0.ogg`, `doublejump_other_1.ogg`, `doublejump_other_2.ogg`, `doublejump_other_3.ogg` |
| `flipreset_local` | flip reset (the car you're watching) | `flipreset_local_0.ogg` |
| `flipreset_other` | flip reset (other cars, 3D) | `flipreset_other_0.ogg` |
| `goal_explosion` | goal scored (event layer, with the explosion) | `goal_explosion_0.ogg`, `goal_explosion_1.ogg`, `goal_explosion_2.ogg` |
| `goal_explosion_default` | goal scored (explosion) | `goal_explosion_default_0.ogg` |
| `goal_horn` | goal horn | `goal_horn_0.ogg` |
| `jump_local` | jump (the car you're watching) | `jump_local_0.ogg`, `jump_local_1.ogg`, `jump_local_2.ogg`, `jump_local_3.ogg` |
| `jump_other` | jump (other cars, 3D) | `jump_other_0.ogg`, `jump_other_1.ogg`, `jump_other_2.ogg`, `jump_other_3.ogg` |
| `land_local` | landing (the car you're watching) | `land_local_0.ogg`, `land_local_1.ogg`, `land_local_2.ogg`, `land_local_3.ogg`, `land_local_4.ogg`, `land_local_5.ogg`, `land_local_6.ogg`, `land_local_7.ogg` |
| `land_other` | landing (other cars, 3D) | `land_other_0.ogg`, `land_other_1.ogg`, `land_other_2.ogg`, `land_other_3.ogg`, `land_other_4.ogg`, `land_other_5.ogg`, `land_other_6.ogg`, `land_other_7.ogg` |
| `pad_pickup` | boost pad pickup | `pad_pickup_0.ogg` |
| `supersonic` | going supersonic | `supersonic_0.ogg`, `supersonic_1.ogg`, `supersonic_2.ogg` |
