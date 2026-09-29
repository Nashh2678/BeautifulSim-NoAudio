"""Rocket-League-style shaders (2026-09 restyle). Replaces the old flat-palette + geometry-shader
wireframe look (the blue/red triangle-edge lines) with a single-pass, per-pixel style:

  ARENA   procedural turf (mowing stripes, noise, white field lines), dark stadium walls with faint
          team-colored hex panels, glowing goal frames + net grid, soft blob shadows for ball + cars.
          No geometry shader any more (it emitted 21 verts per arena triangle every frame).
  CAR     real Octane mesh with a per-face material id (body/trim/tire/rim/glass/lights/chassis/metal),
          team body color, Blinn specular + clear-coat env reflection, wheel glow (flip reset).
  BALL    procedural RL-style ball on a smooth sphere: soccer panel layout (truncated icosahedron =
          spherical Voronoi of 12 + 20 directions), fine honeycomb micro-texture, spec + rim light.
  FX      point-sprite particles (boost flame, sparks, explosions), shockwave rings, sky gradient.

All lighting is done in linear space and written out through the same gamma curve.
"""

COMMON = '''
// Per-map light and sky (maps.py THEMES; the valley's values are the original evening constants).
uniform vec3 uSunDir;      // normalised
uniform vec3 uSunCol;
uniform vec3 uSkyZen;
uniform vec3 uSkyMid;
uniform vec3 uSkyHor;
uniform vec3 uSunGlow;
uniform vec3 uSkyGround;
uniform vec3 uAmb;         // ambient tint multiplier
#define SUN_DIR uSunDir
#define SUN_COL uSunCol

float hash1(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash1(i), hash1(i + vec2(1, 0)), f.x), mix(hash1(i + vec2(0, 1)), hash1(i + vec2(1, 1)), f.x), f.y);
}

// Evening sky (used for the background AND for reflections).
vec3 sky_color(vec3 d) {
    float h = d.z;
    vec3 zen = uSkyZen;
    vec3 mid = uSkyMid;
    vec3 hor = uSkyHor;
    vec3 c = mix(hor, mid, smoothstep(-0.02, 0.28, h));
    c = mix(c, zen, smoothstep(0.28, 0.85, h));
    float sd = max(dot(d, SUN_DIR), 0.0);
    c += uSunGlow * (pow(sd, 8.0) * 0.45 + pow(sd, 90.0) * 1.2);
    c = mix(c, uSkyGround, smoothstep(0.0, -0.12, h));   // below horizon
    return c;
}

vec3 to_srgb(vec3 c) {
    c = c / (1.0 + 0.15 * c);
    return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
}
'''


# Shadow casters (ball + cars) shared by the arena floor and the 3D grass
CASTERS = '''
uniform vec4 casters[9];
uniform vec2 casterFwd[9];
uniform vec3 casterF3[9];      // car forward / up (3D), for the contact occlusion
uniform vec3 casterU3[9];
uniform int nCasters;
float sdRoundBox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}
// Signed distance from world point w to caster i (car: its oriented hitbox, rounded a little; ball: its sphere).
float casterSD(int i, vec3 w) {
    vec3 d = w - casters[i].xyz;
    if (dot(casterFwd[i], casterFwd[i]) > 0.0) {
        vec3 F = casterF3[i], U = casterU3[i];
        vec3 R = cross(U, F);
        vec3 q = vec3(dot(d, F) - 13.88, dot(d, R), dot(d, U) - 20.75);    // Octane hitbox centre offset
        vec3 e = abs(q) - vec3(55.0, 38.0, 14.0);                           // 118 x 84 x 36, minus the rounding
        return length(max(e, 0.0)) + min(max(e.x, max(e.y, e.z)), 0.0) - 4.0;
    }
    return length(d) - 91.25;
}
// Shadow of the ball and the cars on surface point p: a soft shadow ray marched toward the light (mostly straight
// up, a little toward the sun) against the casters' real shapes -- the shadow has the car's outline, follows its
// orientation, lands on the curves too, and gets softer the higher the body is above the surface.
float shadowAt(vec3 p) {
    vec3 L = normalize(mix(SUN_DIR, vec3(0.0, 0.0, 1.0), 0.65));
    float sh = 0.0;
    for (int i = 0; i < nCasters; i++) {
        vec3 dc = casters[i].xyz - p;
        float along = dot(dc, L);
        if (along < -60.0 || along > 1800.0) continue;
        vec3 w = p + L * along;                                          // the light ray's closest point to the caster
        if (dot(w - casters[i].xyz, w - casters[i].xyz) > 190.0 * 190.0) continue;
        // one shape evaluation there: the caster's cross-section seen along the light = its outline (a car's
        // footprint follows its orientation); the edge softens the higher the body is above the surface
        float sd = casterSD(i, w);
        float soft = 3.0 + along * 0.08;
        float strength = dot(casterFwd[i], casterFwd[i]) > 0.0 ? 0.62 : 0.55;
        sh = max(sh, (1.0 - smoothstep(-soft, soft, sd)) * strength * (1.0 - smoothstep(600.0, 1800.0, along)));
    }
    return sh;
}
// Coverage of a line of half-width w (world units) at distance d, for a pixel whose footprint ACROSS the line is
// g world units. Energy-preserving: once the line is thinner than a pixel it keeps a one-pixel width and fades by
// w / w' instead of getting fatter.
float lineCov(float d, float w, float g) {
    float we = max(w, 0.5 * g);
    return clamp((we - abs(d)) / g + 0.5, 0.0, 1.0) * (w / we);
}
// RL's ball marker, inner piece, for a point d (uu) from the ball's x/y: a ring split into 4 by a cross whose gaps
// are ALWAYS the same width; only the ring's size changes with the ball's height t (0 = on the ground .. 1 = high),
// growing thicker as it shrinks until it is a small solid disc. g = pixel footprint (uu) for anti-aliasing.
float markInner(vec2 d, float t, float g) {
    float Ro = 91.25 * mix(0.80, 0.33, t) + mix(2.4, 3.0, t);
    float th = mix(4.8, Ro, smoothstep(0.55, 1.0, t));
    float r = length(d);
    float ring = clamp((Ro - r) / g + 0.5, 0.0, 1.0) * clamp((r - (Ro - th)) / g + 0.5, 0.0, 1.0);
    const float GAP = 3.4;                                      // half gap width (uu), fixed
    float gaps = clamp((min(abs(d.x), abs(d.y)) - GAP) / g + 0.5, 0.0, 1.0);
    return ring * gaps;
}
'''

# --------------------------------------------------------------------------------------------- #
ARENA_VERT = '''
#version 330
uniform mat4 m_vp;
in vec3 in_position;
in vec3 in_normal;
out vec3 v_pos;
out vec3 v_nrm;
out float v_grid;
void main() {
    vec3 p = in_position;
    // The mesh's visible floor (Grid.001) sits at z=-7.1 but RocketSim's ground is z=0: without this
    // every car and the ball float 7 uu above the field.
    v_grid = 0.0;
    if (p.z < -7.0 && p.z > -7.2) { p.z = 0.0; v_grid = 1.0; }
    else if (abs(p.z) < 0.5) p.z -= 2.0;      // mesh's own z=0 geometry would z-fight the floor (goal flicker)
    v_pos = p;
    v_nrm = in_normal;
    gl_Position = m_vp * vec4(p, 1.0);
}
'''

ARENA_FRAG = '''
#version 330
''' + COMMON + '''
uniform vec3 camPos;
''' + CASTERS + '''
uniform vec3 blueCol;
uniform vec3 orangeCol;
uniform int passMode;          // 0 = opaque parts, 1 = translucent walls + ceiling
uniform float time;
uniform float detailBias;      // 1 = smooth distant detail, 2 = "sharp" (detail kept twice as far)
uniform vec4 ballMark;         // RL ball marker: ball x, y, centre z, height factor 0..1 (< 0 = off)
uniform int mapId;             // 0 valley, 1 Forbidden Temple, 2 Parc de Paris, 3 Orbit (maps.py)
uniform vec3 grassCol;         // the turf's base colour on this map
uniform sampler2D bladeTex;    // the blade pattern (BLADE_FRAG), 16 texels per blade cell, 128 cells, repeating
uniform sampler2D grainTex;    // the turf grain (GRAIN_FRAG), repeating every 768 uu
uniform float glassK;          // wall / ceiling glass opacity factor (1 = valley)
uniform int bakeAlbedo;        // 1 = top-down bake of the turf + markings colour (unlit) for the 3D grass blades

// Greek-key (meander) tile, 5x5 cells, row 0 at the bottom (Forbidden Temple's centre band)
const float KEY[25] = float[25](1.,1.,1.,1.,1.,  1.,0.,0.,0.,1.,  1.,0.,1.,1.,1.,  1.,0.,0.,0.,0.,  1.,1.,1.,1.,1.);

in vec3 v_pos;
in vec3 v_nrm;
in float v_grid;
out vec4 f_color;

float aline(float d, float w) {
    float fw = max(fwidth(d), 1e-3);
    return 1.0 - smoothstep(w - fw, w + fw, abs(d));
}
float hexEdge(vec2 p, float s) {
    p /= s;
    vec2 r = vec2(1.0, 1.7320508);
    vec2 h = r * 0.5;
    vec2 a = mod(p, r) - h;
    vec2 b = mod(p - h, r) - h;
    vec2 g = dot(a, a) < dot(b, b) ? a : b;
    g = abs(g);
    return (0.5 - max(dot(g, normalize(r)), g.x)) * s;
}
// inside the playable footprint (walls + 45deg corners + goal mouths)?
float footprint(vec2 p) {
    float field = max(max(abs(p.x) - 4096.0, abs(p.y) - 5120.0), (abs(p.x) + abs(p.y) - 8064.0) * 0.7071);
    float goal = max(abs(p.x) - 900.0, abs(p.y) - 6000.0);
    return min(field, goal);
}
// RL's white ball marker, projected straight down onto the floor / floor-wall curve under the ball: a fixed
// outer ring on the ball's x/y, and an inner ring of 4 arcs that is almost as big as the outer one when the
// ball is on the ground and shrinks (arcs shortening) as it rises, down to 4 dots by ~half the ceiling height.
// The two never touch. Derivatives are taken before any masking (no divergent fwidth).
float ballMarkAt(vec3 p, vec3 n) {
    vec2 d = p.xy - ballMark.xy;
    float r = length(d);
    float t = clamp(ballMark.w, 0.0, 1.0);
    const float RO = 91.25;                                       // the outer ring = the ball's size
    // pixel footprint along the ring's normal (true screen-space gradient length)
    float gr = max(length(vec2(dFdx(r), dFdy(r))), 1e-3);
    float outer = lineCov(r - RO, 2.4, gr);
    float inner = markInner(d, t, gr);
    // any surface facing up at all (floor, the whole floor-wall curve up to where it turns vertical), not above
    // the ball's top: a ball resting against the wall projects onto the curve higher than its centre
    float on = step(0.0, ballMark.w) * step(p.z, ballMark.z + 91.25) * step(0.03, n.z) * step(r, 120.0);
    return max(outer, inner) * on;
}

void main() {
    vec3 p = v_pos;
    vec3 n = normalize(v_nrm);
    vec3 V = normalize(camPos - p);
    float team = smoothstep(-400.0, 400.0, p.y);
    vec3 teamCol = mix(blueCol, orangeCol, team);
    float endness = smoothstep(2500.0, 5120.0, abs(p.y));
    bool inGoal = abs(p.y) > 5135.0;

    // ---- classify ----
    bool grid = v_grid > 0.5;
    // curved floor->wall transition. Height-capped at the real curve top (~260 uu everywhere): the corner
    // mesh has smoothed normals that keep n.z > 0.12 far higher, which made the corner ramp too tall.
    // One height for everything (sides, corners, back walls) so the dark band tops out on a single level line.
    bool ramp = !grid && n.z > -0.5 && p.z < 250.0;
    bool wall = !grid && !ramp && n.z >= -0.12;
    bool ceil = !grid && n.z < -0.12;
    bool translucent = wall || ceil || (ramp && inGoal);          // goal nets see-through like RL
    if ((passMode == 0) == translucent) discard;

    vec3 col; vec3 emis = vec3(0.0); float alpha = 1.0; float spec = 0.0;

    if (grid) {
        // The floor mesh is a big square: cut it where the curved ramps START (not at the wall line),
        // so no turf / slab sticks out under the curves when seen through the glass.
        float inner = max(max(abs(p.x) - 3760.0, abs(p.y) - 4880.0), (abs(p.x) + abs(p.y) - 8064.0) * 0.7071 + 330.0);
        float goalIn = max(abs(p.x) - 880.0, abs(p.y) - 5980.0);
        if (min(inner, goalIn) > 0.0) discard;
        if (inGoal) {
            col = vec3(0.05, 0.055, 0.06) * (0.85 + 0.3 * vnoise(p.xy / 40.0));
        } else {
            // ---- grass: soft large patches, mid clumps, fine blade grain (anisotropic), all
            //      filtered by the pixel footprint so it never shimmers ----
            vec2 q = p.xy;
            float px = max(length(fwidth(q)), 1e-3) / max(detailBias, 1.0);   // uu per pixel
            // uniform base (no patches); individual strands faked as short, randomly oriented blades in
            // two jittered cell layers, each blade darker at its root and bright at its tip. Filtered to the
            // average colour as blades get smaller than a pixel, so it's crisp up close and smooth far away.
            col = grassCol;
            // blades fade over a wider band, on a footprint between the longest axis (no shimmer) and the
            // area (sharp), so there's no visible line where the strands stop
            float pxb = mix(px, sqrt(max(length(dFdx(q)) * length(dFdy(q)), 1e-6)) / max(detailBias, 1.0), 0.5);
            float fineW = 1.0 - smoothstep(0.7, 4.5, pxb);
            if (fineW > 0.0 && mapId != 3) {                 // (Orbit: a metal floor, no blades)
                // the same two blade layers as before, read from the baked pattern (was 18 cells evaluated per pixel)
                float blades = texture(bladeTex, q / (3.2 * 128.0)).r * 0.6
                             + texture(bladeTex, (mat2(0.8, -0.6, 0.6, 0.8) * q + 11.0) / (2.1 * 128.0)).r * 0.4;
                vec3 dark = col * 0.62, lit = col * 1.30 + vec3(0.02, 0.03, 0.0);
                col = mix(col, mix(dark, lit, clamp(blades * 1.4, 0.0, 1.0)), fineW);
            }
            // mid/far texture: clumps (~12 uu) and patches (~30 uu) that keep the turf textured well past
            // where single blades turn sub-pixel, each faded out only once IT gets under ~2 pixels, so the
            // detail thins out gradually with distance instead of the whole field turning flat at once
            // Filtered by the footprint AREA (geometric mean of the two screen axes) instead of its longest
            // axis: at the grazing angles of a pitch the longest axis is many times the short one, and
            // using it smeared the turf flat a few car lengths away (poor man's anisotropic filtering).
            float pxa = sqrt(max(length(dFdx(q)) * length(dFdy(q)), 1e-6)) / max(detailBias, 1.0);
            // 6, 12, 24, 48 uu value-noise octaves, baked (GRAIN_FRAG); the mipmaps fade each octave out as it
            // gets smaller than a few pixels (it was 16 noise evaluations per pixel)
            float grain = (texture(grainTex, q / 768.0).r - 0.5) * 4.0;
            col *= 1.0 + 0.13 * grain * (1.0 - 0.6 * fineW);
            // gentle per-metre value noise so even the averaged far field isn't perfectly flat
            col *= 0.95 + 0.10 * vnoise(q / 40.0);
            col = mix(col, col * (0.80 + 0.4 * teamCol), 0.05 + 0.05 * endness);
            // ---- markings (RL "Champions Field" style), each half in its team colour. Distances along the
            //      field: ay = |y| from the halfway line, g = 5120 - |y| from the goal line. Chevrons are the
            //      level lines of g + |x| (pointing to midfield) or ay +/- |x| (centre ring / centre disc). ----
            // deep "paint" team colours (the lighter blueCol / orangeCol are for glows and trims)
            const vec3 PAINT_B = vec3(0.012, 0.10, 0.78), PAINT_O = vec3(0.82, 0.16, 0.02);
            vec3 tc = q.y < 0.0 ? PAINT_B : PAINT_O;
            // saturation vs the deep paint: 0.65 on both halves (each raised +30% from 0.5, blue then orange)
            tc = mix(vec3(dot(tc, vec3(0.2126, 0.7152, 0.0722))), tc, 0.65);
            float ax = abs(q.x), ay = abs(q.y), g = 5120.0 - ay, r = length(q);
            float fill = 0.0, dark = 0.0, white = 0.0, zone = 0.0;
            // 1) solid box in front of the goal (640 deep, +-1500) with dark ">" chevrons
            float boxD = max(max(-g, g - 640.0), ax - 1500.0);
            float inBox = 1.0 - smoothstep(-1.0, 1.0 + fwidth(boxD), boxD);
            fill = max(fill, inBox);
            dark = max(dark, inBox * aline((fract((g + ax) / 380.0) - 0.5) * 380.0, 9.0));
            // 2) striped zone (goal line -> 1770 out, +-2900): thick chevron bands, thin outline
            float zD = max(max(-g, g - 1770.0), ax - 2900.0);
            float inZone = 1.0 - smoothstep(-1.0, 1.0 + fwidth(zD), zD);
            float bands = aline((fract((g + ax) / 640.0) - 0.5) * 640.0, 150.0);
            fill = max(fill, inZone * bands);
            zone = max(zone, inZone);
            white = max(white, 0.45 * aline(zD, 7.0));
            // 3) the big "D" arc closing the zone toward midfield (circle centred 11510 behind the goal line)
            float arc = length(vec2(q.x, 11510.0 - ay)) - 8660.0;
            white = max(white, aline(arc, 12.0) * step(ax, 2900.0) * step(ay, 3360.0));
            // 4) thin lengthwise team lanes + two white dashed lines
            float lane = 0.0;
            lane = max(lane, (aline(ax - 3000.0, 9.0) + aline(ax - 2000.0, 9.0)) * step(ay, 3350.0));
            lane = max(lane, aline(ax - 1000.0, 9.0) * step(1090.0, ay) * step(ay, 2935.0));
            lane = max(lane, aline(q.x, 9.0) * step(1330.0, ay) * step(ay, 2750.0));
            fill = max(fill, min(lane, 1.0) * 0.9);
            white = max(white, 0.6 * aline(ax - 1130.0, 7.0) * step(1000.0, ay) * step(ay, 3100.0)
                                  * step(fract(ay / 220.0), 0.55));
            // 5) centre: split solid disc (dark ">" lines pointing to the centre line), ring of outward
            //    chevron stripes, white circle, halfway line, centre spot
            float disc = 1.0 - smoothstep(580.0 - fwidth(r), 580.0 + fwidth(r), r);
            fill = max(fill, disc);
            dark = max(dark, disc * aline((fract((ay - ax) / 300.0) - 0.5) * 300.0, 8.0));
            float ring = smoothstep(630.0, 640.0, r) * (1.0 - smoothstep(1020.0, 1030.0, r));
            fill = max(fill, ring * aline((fract((ay + ax) / 320.0) - 0.5) * 320.0, 55.0));
            zone = max(zone, (1.0 - smoothstep(1070.0, 1080.0, r)) * 0.7);
            white = max(white, aline(r - 1080.0, 11.0));
            white = max(white, aline(q.y, 11.0));
            white = max(white, 1.0 - smoothstep(34.0, 38.0 + fwidth(r), r));
            dark = max(dark, aline(q.y, 22.0) * disc);              // thin dark split between the half discs
            // ---- per-map field style (the valley keeps the markings above as they are) ----
            if (mapId == 1) {                        // Forbidden Temple: zones and lanes only faintly tinted
                fill *= 0.12 + 0.88 * (1.0 - smoothstep(1000.0, 1100.0, r)); dark *= 0.5; zone *= 0.4;
            } else if (mapId == 2) {                 // Parc de Paris: striped turf, white lines, a small centre disc
                float pw = aline(q.y, 11.0) * step(0.5, fract(q.x / 300.0 + 0.25));         // dashed halfway line
                pw = max(pw, aline(r - 1080.0, 11.0));
                pw = max(pw, 0.8 * aline(r - 2900.0, 10.0));
                pw = max(pw, aline(zD, 10.0));
                pw = max(pw, aline(boxD, 9.0));
                pw = max(pw, aline(arc, 12.0) * step(ax, 2900.0) * step(ay, 3360.0));
                pw = max(pw, 0.7 * aline(ax - 3000.0, 8.0) * step(ay, 3350.0));
                white = pw;
                fill = 1.0 - smoothstep(160.0 - fwidth(r), 160.0 + fwidth(r), r);
                dark = 0.0; zone = 0.0;
                col *= 0.88 + 0.20 * step(0.5, fract(r / 760.0));                           // mowing rings
            }
            if (mapId == 3) {
                // Orbit: dark metal deck plates, a faint cyan light grid, every marking a neon line
                vec2 pl = q / 512.0;
                vec2 pf = abs(fract(pl) - 0.5);
                float seam = aline((0.5 - max(pf.x, pf.y)) * 512.0, 3.0);
                col = grassCol * (0.85 + 0.3 * hash1(floor(pl))) * (1.0 + 0.10 * grain);
                col *= 1.0 - 0.55 * seam;
                float lg = max(aline((fract(q.x / 1024.0) - 0.5) * 1024.0, 4.0), aline((fract(q.y / 1024.0) - 0.5) * 1024.0, 4.0));
                emis += vec3(0.10, 0.35, 0.60) * lg * 0.22;
                float fz = fill * (0.25 + 0.75 * (1.0 - smoothstep(1000.0, 1100.0, r)));      // centre full, zones faint
                col = mix(col, tc * 0.5, fz * 0.25);
                emis += tc * fz * (1.0 - dark) * 0.12;
                emis += mix(tc * 1.8, vec3(0.75, 0.92, 1.0), 0.35) * white * 0.9;
                col = mix(col, vec3(0.9), white * 0.35);
                spec = 0.0;                                   // matte deck (was a glossy 0.30 sheen)
            } else {
                // paint: zones sit on slightly darker turf, team colour over it, dark chevrons over solid fills
                col *= 1.0 - 0.18 * zone;
                col = mix(col, tc, fill * 0.92);
                col = mix(col, col * 0.25, dark * fill);
                col = mix(col, vec3(0.80, 0.83, 0.80), white * 0.8);
                emis += tc * fill * (1.0 - dark) * 0.12;
                spec = 0.03;
            }
            if (mapId == 1) {
                // centre medallion (cream disc, red ring, dark red core) and a Greek-key band around it in the
                // team colours, like the temple's floor
                float med = 1.0 - smoothstep(430.0 - fwidth(r), 430.0 + fwidth(r), r);
                col = mix(col, vec3(0.72, 0.64, 0.54), med * 0.95);
                col = mix(col, vec3(0.50, 0.05, 0.03), aline(r - 305.0, 28.0) * med);
                col = mix(col, vec3(0.28, 0.035, 0.03), 1.0 - smoothstep(150.0, 150.0 + fwidth(r), r));
                col = mix(col, vec3(0.80, 0.70, 0.45), aline(r - 450.0, 8.0));
                float s_ = atan(q.y, q.x) * 660.0;
                vec2 kt = vec2(fract(s_ / 200.0), (r - 560.0) / 200.0);
                if (kt.y > 0.0 && kt.y < 1.0) {
                    ivec2 kc = ivec2(floor(kt * 5.0));
                    float key = KEY[kc.y * 5 + kc.x] * (1.0 - smoothstep(0.35, 0.8, fwidth(s_) / 40.0));
                    col = mix(col, tc * 1.05, key * 0.9);
                    emis += tc * key * 0.10;
                }
            }
        }
    } else if (inGoal) {
        // goal net: see-through like the walls, team-coloured net grid
        vec2 nq = abs(n.y) > 0.5 ? vec2(p.x, p.z) : (abs(n.x) > 0.5 ? vec2(p.y, p.z) : p.xy);
        float g = max(aline(fract(nq.x / 70.0) - 0.5, 0.045), aline(fract(nq.y / 70.0) - 0.5, 0.045));
        col = mix(vec3(0.05, 0.055, 0.07), teamCol, 0.25);
        alpha = 0.10 + 0.55 * g;
        emis += teamCol * 0.45 * g;
    } else if (ramp) {
        // floor->wall curve in the colour of the team whose half it is (switches at the halfway line),
        // brighter toward its top edge, with a glowing rim like RL's arena boards
        vec3 rc = mix(vec3(0.012, 0.10, 0.78), vec3(0.82, 0.16, 0.02), smoothstep(-30.0, 30.0, p.y));
        rc = mix(vec3(dot(rc, vec3(0.2126, 0.7152, 0.0722))), rc, 0.7);     // 30% less saturated
        float h = clamp(p.z / 250.0, 0.0, 1.0);
        col = rc * (0.55 + 0.35 * h) * (0.92 + 0.08 * step(0.5, fract(p.z / 55.0)));
        emis += rc * (0.10 + 0.25 * h) + mix(rc, vec3(1.0), 0.35) * 0.8 * aline(p.z - 244.0, 5.0);
        spec = 0.15;
    } else {
        // ---- translucent glass walls / ceiling with hex panels ----
        // One continuous hex pattern over walls + ceiling: "unfold" the shell onto the ceiling plane by
        // pushing each point outward (toward its nearest wall) by its drop below the ceiling. Position-
        // based (not normal-based: the mesh normals are too coarse), continuous and never folds over, so
        // the ceiling->wall curve no longer stretches the hexes into a "waterfall".
        // The push direction BLENDS smoothly across the side-wall / 45-degree-corner / back-wall seams
        // (softmax of the face distances): a hard switch between face normals tore the pattern apart
        // along every vertical seam. (No flat unfolding of this shell is seam-free; blending spreads
        // the unavoidable mismatch over ~1000 uu instead of a visible break.)
        vec2 ap = abs(p.xy);
        float dS = ap.x - 4096.0, dB = ap.y - 5120.0, dC = (ap.x + ap.y - 8064.0) * 0.7071;
        float dm = max(dS, max(dB, dC));
        float wS = exp((dS - dm) / 260.0), wB = exp((dB - dm) / 260.0), wC = exp((dC - dm) / 260.0);
        vec2 outw = normalize(wS * vec2(1.0, 0.0) + wB * vec2(0.0, 1.0) + wC * vec2(0.7071, 0.7071));
        outw *= vec2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
        vec2 wq = p.xy + outw * max(2068.0 - p.z, 0.0);
        float he = hexEdge(wq, 260.0);
        float hl = aline(he, 5.0);
        col = mix(vec3(0.10, 0.12, 0.16), teamCol, 0.35 + 0.35 * endness);
        float fres = pow(1.0 - abs(dot(n, V)), 3.0);
        alpha = (0.13 + 0.12 * fres) * glassK + hl * 0.55 * (0.5 + 0.5 * glassK);   // no floor-level glow band
        emis += teamCol * hl * (0.6 + 0.3 * endness);
        if (ceil) alpha *= 0.85;
        // glowing rounded goal frame around the goal mouth on the back walls
        if (abs(abs(p.y) - 5120.0) < 40.0 && abs(n.y) > 0.6) {
            float d = sdRoundBox(vec2(p.x, p.z - 325.8), vec2(898.6, 325.8), 72.0);
            float frame = aline(d - 24.0, 24.0) * step(0.0, d);
            emis += teamCol * 2.4 * frame;
            col = mix(col, vec3(1.0), frame * 0.5);
            alpha = max(alpha, frame * 0.95);
        }
    }

    if (bakeAlbedo == 1) { f_color = vec4(col, grid && !inGoal ? 1.0 : 0.0); return; }
    float bmk = ballMarkAt(p, n) * (ceil ? 0.0 : 1.0);
    col = mix(col, vec3(0.85), bmk * 0.85);
    emis = mix(emis, vec3(0.40), bmk);
    alpha = max(alpha, bmk * 0.9);                                // on the glass part of the curve too

    // ---- lighting ----
    float ndl = max(dot(n, SUN_DIR), 0.0);
    vec3 amb = mix(vec3(0.12, 0.11, 0.12), vec3(0.42, 0.40, 0.46), n.z * 0.5 + 0.5) * uAmb;
    float sh = (passMode == 0 && (grid || ramp) && !inGoal) ? shadowAt(p) : 0.0;
    float ao = 0.0;
    vec3 lit = col * (amb * 1.25 + SUN_COL * ndl * 0.75 * (1.0 - sh)) * (1.0 - ao);
    vec3 H = normalize(SUN_DIR + V);
    lit += SUN_COL * spec * pow(max(dot(n, H), 0.0), 30.0) * (1.0 - sh);
    lit += emis;
    float dist = length(camPos - p);
    lit = mix(lit, vec3(0.30, 0.20, 0.22), smoothstep(5000.0, 16000.0, dist) * 0.35);
    f_color = vec4(to_srgb(lit) * (passMode == 1 ? alpha : 1.0), alpha);   // premultiplied in pass 1
}
'''

# The arena pass variants, compiled separately: with passMode a constant the compiler drops the other pass's code (the
# glass shader carries no turf / markings / shadow code and vice versa) -- fewer registers, better GPU occupancy.
ARENA_FRAG_OPAQUE = ARENA_FRAG.replace("uniform int passMode;", "const int passMode = 0;")
ARENA_FRAG_GLASS = ARENA_FRAG.replace("uniform int passMode;", "const int passMode = 1;")

# --------------------------------------------------------------------------------------------- #
# 3D grass: real blades (one tapered, bent triangle each) on the field near the camera. Fully generated on the GPU:
# one instance per 128 uu field tile (main.py culls tiles to the view and sorts them into a few blade-count buckets
# by distance); gl_VertexID picks the blade. Blade k of a tile sits at the k-th point of the R2 low-discrepancy
# sequence (shifted per tile), so ANY prefix of a tile's blades is evenly spread: thinning with distance just drops
# the last blades of the prefix, each shrinking to nothing over a band -- no popping, no visible LOD rings. The
# colour comes from a top-down bake of the floor shader (turf + markings), so blades on a line are line-coloured.
GRASS_VERT = '''
#version 330
''' + COMMON + CASTERS + '''
uniform mat4 m_vp;
uniform vec3 camPos;
uniform float time;
uniform float tileSize;
uniform float density0;     // blades per tile at full density
uniform float nearD;        // full density within this distance of the camera; ~1/d^2 beyond (constant on screen)
uniform float farD;         // no blades past this
uniform float bladeH;       // mean blade height (uu)
uniform vec4 ballMark;      // as the arena shader: ball x, y, centre z, height factor (< 0 = off)
uniform sampler2D albedoTex;
uniform sampler2D padMask;  // 1 = grass, 0 = a boost pad's footprint
in vec2 i_tile;             // tile min corner
out vec3 v_col;
out vec3 v_nrm;
out vec3 v_pos;
out float v_t;              // 0 root .. 1 tip
out float v_light;          // (1 - sun shadow) at the root
out float v_ao;

float markRoot(vec2 q) {                    // RL ball marker at a blade root (no derivatives: blades are discrete)
    if (ballMark.w < 0.0 || ballMark.z > 1400.0) return 0.0;
    vec2 d = q - ballMark.xy;
    float r = length(d);
    if (r > 100.0) return 0.0;
    float outer = step(abs(r - 91.25), 3.2);
    return max(outer, step(0.5, markInner(d, clamp(ballMark.w, 0.0, 1.0), 1.5)));
}

void main() {
    int blade = gl_VertexID / 3;
    int k = gl_VertexID - blade * 3;
    vec2 seed = vec2(hash1(i_tile * 0.0131 + 0.7), hash1(i_tile * 0.0173 + 5.3));
    vec2 f = fract(seed + vec2(0.7548776662, 0.5698402910) * float(blade + 1));
    // whole tile outside the view (bounding sphere vs the clip volume, with a generous margin): skip it
    vec4 tc = m_vp * vec4(i_tile + 0.5 * tileSize, 0.0, 1.0);
    float tr = 0.75 * tileSize + 20.0;
    if (tc.w < -tr || abs(tc.x) > tc.w + 3.0 * tr || abs(tc.y) > tc.w + 3.0 * tr) {
        v_col = vec3(0.0); v_nrm = vec3(0.0, 0.0, 1.0); v_pos = vec3(0.0); v_t = 0.0; v_light = 1.0; v_ao = 0.0;
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    vec2 root = i_tile + f * tileSize;
    float d = length(camPos - vec3(root, 0.0));
    float want = density0 * min(1.0, nearD * nearD / max(d * d, 1.0));
    float keep = clamp((want - float(blade)) / max(0.3 * want, 1.0), 0.0, 1.0);
    keep *= 1.0 - smoothstep(0.65 * farD, farD, d);
    // only on the turf the floor shader keeps (its cut where the ramps start), a little inside it
    float inner = max(max(abs(root.x) - 3760.0, abs(root.y) - 4880.0), (abs(root.x) + abs(root.y) - 8064.0) * 0.7071 + 330.0);
    keep *= 1.0 - smoothstep(-40.0, -10.0, inner);
    if (keep > 0.0)                             // (texture fetch only for blades that survived the thinning)
        keep *= smoothstep(0.25, 0.75, textureLod(padMask, root / vec2(8400.0, 10400.0) + 0.5, 0.0).r);
    if (keep <= 0.0) {                          // thinned-out blade: nothing else to compute
        v_col = vec3(0.0); v_nrm = vec3(0.0, 0.0, 1.0); v_pos = vec3(0.0); v_t = 0.0; v_light = 1.0; v_ao = 0.0;
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    float r1 = hash1(root * 1.37 + 0.1), r2 = hash1(root * 2.11 + 7.0), r3 = hash1(root * 0.73 + 3.0);
    float h = bladeH * (0.55 + 0.9 * r1 * r1) * keep;
    // pressed flat under the cars' wheels / body and under a ball sitting on the turf
    float ao = 0.0, lit = 1.0;
    bool nearCaster = false;
    for (int i = 0; i < nCasters; i++) {
        vec4 c = casters[i];
        vec2 dd = root - c.xy;
        if (dot(dd, dd) > 170.0 * 170.0 || c.z > 130.0) continue;           // nothing to flatten here
        nearCaster = true;
        vec2 fw = casterFwd[i];
        if (dot(fw, fw) > 0.0) {
            vec2 q = vec2(dot(dd, fw), dot(dd, vec2(-fw.y, fw.x)));
            float sd = sdRoundBox(q - vec2(4.0, 0.0), vec2(58.0, 40.0), 20.0);
            h *= mix(1.0, 0.25, (1.0 - smoothstep(-2.0, 10.0, sd)) * (1.0 - smoothstep(24.0, 40.0, c.z)));
        } else {
            h *= mix(1.0, 0.3, (1.0 - smoothstep(20.0, 55.0, length(dd))) * (1.0 - smoothstep(96.0, 110.0, c.z)));
        }
    }
    vec3 rp = vec3(root, 0.0);
    lit = 1.0 - shadowAt(rp);
    ao = 0.0;

    float yaw = r2 * 6.2831853;
    vec2 side = vec2(cos(yaw), sin(yaw));
    vec2 across = vec2(-side.y, side.x);
    float wind = sin(time * 1.6 + root.x * 0.004 + root.y * 0.006) * 0.5 + sin(time * 2.9 + root.y * 0.011) * 0.25;
    vec2 lean = across * (r3 - 0.5) * 1.1 + vec2(0.6, 0.35) * wind * 0.35;
    float w = (0.55 + 0.5 * r3) * (1.0 + d / 450.0);          // farther blades are wider: the same coverage from fewer blades
    vec3 tip = vec3(root + lean * h, h * (1.0 - 0.25 * dot(lean, lean)));
    vec3 P = k == 2 ? tip : vec3(root + side * w * (k == 0 ? -0.5 : 0.5), 0.0);
    v_pos = P;
    v_t = k == 2 ? 1.0 : 0.0;
    v_nrm = normalize(cross(tip - rp, vec3(side, 0.0)));
    vec3 alb = textureLod(albedoTex, root / vec2(8400.0, 10400.0) + 0.5, 0.0).rgb;
    v_col = alb * (0.85 + 0.3 * r1);
    v_light = lit;
    v_ao = ao;
    gl_Position = m_vp * vec4(P, 1.0);
    if (h < 0.05) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // culled blade: a degenerate point off screen
}
'''

GRASS_FRAG = '''
#version 330
''' + COMMON + '''
uniform vec3 camPos;
in vec3 v_col;
in vec3 v_nrm;
in vec3 v_pos;
in float v_t;
in float v_light;
in float v_ao;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_nrm);
    vec3 V = normalize(camPos - v_pos);
    if (dot(n, V) < 0.0) n = -n;
    // lit mostly like the floor under it (so the far, thinning blades melt into the turf), a little by the blade
    n = normalize(mix(vec3(0.0, 0.0, 1.0), n, 0.4));
    float ndl = max(dot(n, SUN_DIR), 0.0);
    vec3 amb = mix(vec3(0.12, 0.11, 0.12), vec3(0.42, 0.40, 0.46), n.z * 0.5 + 0.5) * uAmb;
    vec3 col = v_col * mix(0.62, 1.22, v_t) + vec3(0.012, 0.018, 0.0) * v_t;       // dark roots, sunlit tips
    vec3 lit = col * (amb * 1.25 + SUN_COL * ndl * 0.75 * v_light) * (1.0 - v_ao);
    // light through the blade tips when looking toward the sun
    lit += col * SUN_COL * 0.25 * v_t * v_light * pow(max(dot(-V, SUN_DIR), 0.0), 4.0);
    f_color = vec4(to_srgb(lit), 1.0);
}
'''

# --------------------------------------------------------------------------------------------- #
CAR_VERT = '''
#version 330
uniform mat4 m_vp;
uniform mat4 m_model;
in vec3 in_position;
in vec3 in_normal;
in float in_mat;
out vec3 v_pos;
out vec3 v_nrm;
flat out int v_mat;
void main() {
    vec4 wp = m_model * vec4(in_position, 1.0);
    v_pos = wp.xyz;
    v_nrm = mat3(m_model) * in_normal;
    v_mat = int(in_mat + 0.5);
    gl_Position = m_vp * wp;
}
'''

CAR_FRAG = '''
#version 330
''' + COMMON + '''
uniform vec3 camPos;
uniform vec3 bodyCol;       // team primary (linear)
uniform float wheelGlow;    // 0..1 flip-reset flash on the tires
uniform vec3 glowCol;
uniform float brakeLight;   // 0..1
in vec3 v_pos;
in vec3 v_nrm;
flat in int v_mat;
out vec4 f_color;

void main() {
    vec3 n = normalize(v_nrm);
    vec3 V = normalize(camPos - v_pos);
    if (dot(n, V) < 0.0) n = -n;                    // thin double-sided parts (wing, fins)
    vec3 base; float ks; float gloss; float env; vec3 emis = vec3(0.0);
    if (v_mat == 0)      { base = bodyCol;                   ks = 0.32; gloss = 55.0;  env = 0.22; }
    else if (v_mat == 1) { base = vec3(0.028, 0.029, 0.032); ks = 0.35; gloss = 35.0;  env = 0.25; }
    else if (v_mat == 2) { base = vec3(0.018, 0.018, 0.019); ks = 0.06; gloss = 10.0;  env = 0.02;
                           emis = glowCol * wheelGlow * 1.4; }
    else if (v_mat == 3) { base = vec3(0.50, 0.52, 0.56);    ks = 0.60; gloss = 70.0;  env = 0.40;
                           emis = glowCol * wheelGlow * 0.8; }
    else if (v_mat == 4) { base = vec3(0.010, 0.012, 0.016); ks = 0.0;  gloss = 1.0;   env = 0.0;  }  // glass: fully matte
    else if (v_mat == 5) { base = vec3(0.9);                 ks = 0.3;  gloss = 60.0;  env = 0.2;
                           emis = vec3(1.0, 0.95, 0.80) * 1.6; }
    else if (v_mat == 6) { base = vec3(0.5, 0.02, 0.02);     ks = 0.3;  gloss = 60.0;  env = 0.2;
                           emis = vec3(1.0, 0.06, 0.03) * (0.8 + 1.4 * brakeLight); }
    else if (v_mat == 7) { base = vec3(0.020, 0.020, 0.022); ks = 0.10; gloss = 12.0;  env = 0.05; }
    else                 { base = vec3(0.30, 0.31, 0.33);    ks = 0.45; gloss = 50.0;  env = 0.25; }

    float ndl = max(dot(n, SUN_DIR), 0.0);
    vec3 amb = mix(vec3(0.07, 0.07, 0.07), vec3(0.32, 0.35, 0.42), n.z * 0.5 + 0.5) * uAmb;
    vec3 c = base * (amb * 1.5 + SUN_COL * ndl * 1.1);
    vec3 H = normalize(SUN_DIR + V);
    c += SUN_COL * ks * pow(max(dot(n, H), 0.0), gloss) * (gloss + 8.0) / 60.0;
    float fres = 0.04 + 0.96 * pow(1.0 - max(dot(n, V), 0.0), 5.0);
    c += sky_color(reflect(-V, n)) * env * mix(0.25, 1.0, fres);
    c += emis;
    f_color = vec4(to_srgb(c), 1.0);
}
'''

# --------------------------------------------------------------------------------------------- #
BALL_VERT = '''
#version 330
uniform mat4 m_vp;
uniform mat4 m_model;
in vec3 in_position;
out vec3 v_pos;
out vec3 v_obj;
out vec3 v_nrm;
void main() {
    vec4 wp = m_model * vec4(in_position, 1.0);
    v_pos = wp.xyz;
    v_obj = normalize(in_position);
    v_nrm = mat3(m_model) * v_obj;
    gl_Position = m_vp * wp;
}
'''

BALL_FRAG = '''
#version 330
''' + COMMON + '''
uniform vec3 camPos;
uniform float inGoal;      // 1 = ball centre inside a goal mouth, overlapping the goal line
uniform vec4 blurRot;      // spin motion blur: object-space axis (xyz), angle turned over the last frame (w)
in vec3 v_pos;
in vec3 v_obj;
in vec3 v_nrm;
out vec4 f_color;

float hexEdgeB(vec2 p, float s) {
    p /= s;
    vec2 r = vec2(1.0, 1.7320508);
    vec2 h = r * 0.5;
    vec2 a = mod(p, r) - h;
    vec2 b = mod(p - h, r) - h;
    vec2 g = dot(a, a) < dot(b, b) ? a : b;
    g = abs(g);
    return 0.5 - max(dot(g, normalize(r)), g.x);
}

float ballSeam(vec3 o, float k) {
    // signed ~angular distance to the seam curve = sphere intersected with the saddle z = k(x^2 - y^2)
    // (one closed curve that splits the sphere into two interlocking halves, like a tennis ball)
    float f = o.z - k * (o.x * o.x - o.y * o.y);
    vec3 g = vec3(-2.0 * k * o.x, 2.0 * k * o.y, 1.0);
    g -= o * dot(g, o);                                    // tangential gradient
    return f / max(length(g), 1e-3);
}

// Ball surface: two interlocking panels (pearl white / cool grey) split by one continuous recessed seam with a
// muted blue edge on one side and orange on the other, contour grooves following the seam, fine honeycomb
// micro-texture. No pentagons, no lights. o = unit object-space direction.
void ballSurf(vec3 o, out vec3 base, out vec3 ballEmis, out float ks, out float channel) {
    // two interlocking seams (the second = the first rotated a quarter turn onto another axis)
    float d1 = ballSeam(o, 0.95);
    float d2 = ballSeam(o.yzx, 0.95);
    bool near1 = abs(d1) < abs(d2);
    float ds = near1 ? d1 : d2;                            // signed distance to the nearest seam
    float ad = abs(ds);
    float fw = max(fwidth(ad), 1e-4);
    channel = 1.0 - smoothstep(0.026 - fw, 0.026 + fw, ad);          // black seam band
    float edge = smoothstep(0.026 - fw, 0.026 + fw, ad) * (1.0 - smoothstep(0.040 - fw, 0.040 + fw, ad));
    float lip = 1.0 - smoothstep(0.040, 0.10, ad);                          // panel edge rolls into the seam
    // three thin grooves following each seam
    float cx = ad * 18.0;
    float cfw = max(fwidth(cx), 1e-4);
    float contour = (1.0 - smoothstep(0.08 - cfw, 0.08 + cfw, abs(fract(cx) - 0.5)))
                  * step(0.20, ad) * step(ad, 0.37) * (1.0 - smoothstep(0.35, 0.8, cfw));
    // honeycomb + mottled grain (tri-planar on the unit sphere) so the shell never reads as flat cream
    vec3 bw = pow(abs(o), vec3(4.0)); bw /= (bw.x + bw.y + bw.z);
    float hx = hexEdgeB(o.yz * 91.0, 2.6) * bw.x + hexEdgeB(o.xz * 91.0, 2.6) * bw.y + hexEdgeB(o.xy * 91.0, 2.6) * bw.z;
    float hfw = max(fwidth(hx), 1e-4);
    float cellWall = 1.0 - smoothstep(0.04 - hfw, 0.08 + hfw, hx);
    float hfade = 1.0 - smoothstep(0.2, 0.5, hfw * 3.0);
    vec3 so = o * 91.0;
    float mott = (vnoise(so.yz / 9.0) * bw.x + vnoise(so.xz / 9.0 + 7.0) * bw.y + vnoise(so.xy / 9.0 + 13.0) * bw.z) * 0.6
               + (vnoise(so.yz / 3.0) * bw.x + vnoise(so.xz / 3.0 + 3.0) * bw.y + vnoise(so.xy / 3.0 + 5.0) * bw.z) * 0.4;
    float gfw = max(fwidth(so.x / 3.0), 1e-4);
    float grain = mix(0.5, mott, 1.0 - smoothstep(0.5, 1.2, gfw));

    vec3 shell = vec3(0.70, 0.705, 0.72);                                   // very light grey
    shell *= 0.86 + 0.28 * grain;                                           // visible mottling
    shell *= 1.0 - 0.18 * cellWall * hfade;
    shell *= 1.0 - 0.30 * contour;
    shell *= 1.0 - 0.22 * lip;
    // small coloured stripes: every seam blue on one side, orange on the other
    vec3 accent = ds > 0.0 ? vec3(0.15, 0.50, 1.0) : vec3(1.0, 0.50, 0.10);
    base = mix(shell, accent, edge);
    base = mix(base, vec3(0.02, 0.021, 0.025), channel);
    ballEmis = accent * edge * 0.12;
    ks = 0.35 * (1.0 - channel) * (1.0 - 0.4 * lip);
}

vec3 rotAxis(vec3 v, vec3 k, float a) {
    float c = cos(a), s_ = sin(a);
    return v * c + cross(k, v) * s_ + k * dot(k, v) * (1.0 - c);
}

void main() {
    // Spin motion blur: the surface averaged over the rotation of the last frame (blurRot = object-space axis,
    // angle). A fast spin at a modest frame rate otherwise makes the thin high-contrast seams jump from frame
    // to frame (it read as the rotation running at 30 fps); at high fps the angle is tiny and this is a no-op.
    vec3 o0 = normalize(v_obj);
    int ns = blurRot.w > 0.02 ? 6 : 1;
    vec3 base = vec3(0.0), ballEmis = vec3(0.0); float ks = 0.0, channel = 0.0;
    for (int k = 0; k < ns; k++) {
        float a = ns > 1 ? blurRot.w * float(k) / float(ns - 1) : 0.0;
        vec3 b_, e_; float s_, c_;
        ballSurf(rotAxis(o0, blurRot.xyz, a), b_, e_, s_, c_);
        base += b_; ballEmis += e_; ks += s_; channel += c_;
    }
    float inv = 1.0 / float(ns);
    base *= inv; ballEmis *= inv; ks *= inv; channel *= inv;
    float rough = 56.0;

    vec3 n = normalize(v_nrm);
    vec3 V = normalize(camPos - v_pos);
    float ndl = max(dot(n, SUN_DIR), 0.0);
    vec3 amb = mix(vec3(0.10, 0.10, 0.11), vec3(0.40, 0.41, 0.44), n.z * 0.5 + 0.5) * uAmb;
    vec3 c = base * (amb * 1.35 + SUN_COL * ndl * 1.0);
    vec3 H = normalize(SUN_DIR + V);
    c += SUN_COL * ks * pow(max(dot(n, H), 0.0), rough);
    // neutral studio reflection (no sunset horizon band -> no orange streak across the ball)
    vec3 R = reflect(-V, n);
    vec3 env = mix(vec3(0.10, 0.10, 0.11), vec3(0.34, 0.35, 0.38), smoothstep(-0.3, 0.6, R.z));
    float fres = pow(1.0 - max(dot(n, V), 0.0), 5.0);
    c += env * (0.04 + 0.14 * fres) * (1.0 - channel);
    c += ballEmis;

    // ---- goal line: ONLY once the ball centre is inside the goal mouth and touching the line:
    // the part of the ball past the line goes dark (its own skin at 12% light, texture still readable),
    // white seam at the line (RL's goal-line cue) ----
    if (inGoal > 0.5) {
        float dy = abs(v_pos.y) - 5124.25;
        float lw = max(fwidth(dy), 0.01) * 1.5;
        float past = smoothstep(-lw, lw, dy);
        c = mix(c, c * 0.12, past);
        c += vec3(1.0) * (1.0 - smoothstep(0.0, 2.2 + lw, abs(dy))) * 1.3;
    }
    f_color = vec4(to_srgb(c), 1.0);
}
'''

# --------------------------------------------------------------------------------------------- #
SKY_VERT = '''
#version 330
out vec2 v_ndc;
void main() {
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_ndc = pos * 2.0 - 1.0;
    gl_Position = vec4(v_ndc, 1.0, 1.0);        // exactly the far plane; drawn with depth func <=
}
'''

SKY_FRAG = '''
#version 330
''' + COMMON + '''
uniform mat4 invVP;
uniform float time;
uniform int mapId;         // 0 valley, 1 temple, 2 paris, 3 orbit
uniform vec4 cloudA;       // cloud colour lit (rgb), coverage threshold (a: lower = more clouds)
uniform vec4 cloudB;       // cloud colour shadowed (rgb), opacity (a)
uniform float starK;       // faint stars in the darker upper sky (dusk maps)
uniform vec2 cloudShape;   // cloud noise scale along x / y (equal = puffy, very unequal = long wisps)
uniform samplerCube spaceCube;   // the baked static space sky (spaceStatic) + open-sky mask in alpha
uniform int skyBake;       // >= 0: bake face skyBake (0..5 = +X -X +Y -Y +Z -Z) of spaceStatic; -1 = normal
uniform float bakeN;       // cube face size in texels
in vec2 v_ndc;
out vec4 f_color;

// skyline: two layers of wide, stepped towers around the horizon with window grids
vec3 skyline(vec3 d, vec3 c) {
    float az = atan(d.y, d.x) + 3.14159265;
    float el = asin(clamp(d.z, -1.0, 1.0));
    for (int layer = 0; layer < 2; layer++) {
        float cw = layer == 0 ? 0.11 : 0.075;               // tower angular slot width
        float off = layer == 0 ? 0.0 : 0.41;
        float x = (az + off) / cw;
        float cell = floor(x);
        float fx = fract(x);
        float r = hash1(vec2(cell, float(layer) * 7.0));
        float r2 = hash1(vec2(cell * 1.7, float(layer) + 3.0));
        float w0 = 0.12 + 0.18 * r2, w1 = 0.88 - 0.12 * hash1(vec2(cell, 9.0));
        if (fx < w0 || fx > w1 || r < 0.12) continue;
        float bx = (fx - w0) / (w1 - w0);                   // 0..1 across the building
        float top = (layer == 0 ? 0.24 : 0.20) + (layer == 0 ? 0.20 : 0.12) * r * r;
        // Most roofs are flat; ~1 in 4 gets a single WIDE setback (upper block = 70% of the width,
        // barely taller) -- no narrow spires.
        float tier = step(0.75, r2);
        float topHere = top - tier * 0.018 * step(0.7, abs(bx - 0.5) * 2.0);
        if (el > topHere) continue;
        vec3 bcol = layer == 0 ? vec3(0.030, 0.030, 0.045) : vec3(0.10, 0.075, 0.12);
        // window grid in building-local coordinates
        vec2 wg = vec2(bx * (10.0 + floor(r2 * 8.0)), el / 0.0075);
        vec2 wi = floor(wg);
        vec2 wf = fract(wg);
        float frame = step(0.25, wf.x) * step(wf.x, 0.8) * step(0.3, wf.y) * step(wf.y, 0.85);
        float lit = step(0.55, hash1(wi + cell * 13.1)) * frame;
        vec3 win = mix(vec3(1.0, 0.78, 0.48), vec3(0.62, 0.8, 1.0), step(0.78, hash1(wi * 1.7 + cell)));
        bcol += win * lit * (layer == 0 ? 0.75 : 0.30);
        bcol += vec3(0.05, 0.04, 0.07) * (1.0 - bx) * 0.5;  // subtle side shading
        if (r > 0.8 && el > topHere - 0.003 && abs(bx - 0.5) < 0.04)
            bcol += vec3(1.0, 0.1, 0.05) * (0.5 + 0.5 * sin(time * 3.0 + cell));   // roof beacon
        return bcol;
    }
    return c;
}

float fbm3(vec2 p) { return vnoise(p) * 0.5 + vnoise(p * 2.1 + 3.1) * 0.3 + vnoise(p * 4.3 + 7.7) * 0.2; }
float hash3(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }

// Stars: the view direction is projected on the face of a cube (2D grid per face, so every star is a clean disc),
// one candidate per cell (`prob` of the cells hold one), the 3x3 neighbour cells are checked so a star is never cut
// by a cell edge. Sized in pixels (a crisp core >= ~1 px + a faint halo); mostly dim, a few bright; blue, white,
// yellow and orange shades. At 1080p a face cell is ~800/scale px.
vec3 starTint(float h) {
    float tt = fract(h * 13.0);
    return tt < 0.30 ? vec3(0.72, 0.82, 1.0) : (tt < 0.72 ? vec3(1.0, 0.98, 0.95)
         : (tt < 0.90 ? vec3(1.0, 0.88, 0.66) : vec3(1.0, 0.68, 0.52)));
}
vec3 stars(vec3 d, float scale, float prob, float seed) {
    vec3 ad = abs(d);
    vec2 uv; float face;
    if (ad.x >= ad.y && ad.x >= ad.z) { uv = d.yz / ad.x; face = d.x > 0.0 ? 0.0 : 1.0; }
    else if (ad.y >= ad.z)            { uv = d.xz / ad.y; face = d.y > 0.0 ? 2.0 : 3.0; }
    else                              { uv = d.xy / ad.z; face = d.z > 0.0 ? 4.0 : 5.0; }
    vec2 p = uv * scale;
    vec2 fw = fwidth(p);
    float px = max(max(fw.x, fw.y), 1e-4);                     // cells per pixel
    if (px > 0.6) return vec3(0.0);                             // (a face seam in this pixel quad)
    // one cell only: the star's centre stays in the middle 40% of its cell, so with cells of >= ~6 px it is never
    // cut by a cell edge (the 3x3 neighbour search cost 9x as much)
    vec2 c = floor(p);
    float fs = seed + face * 131.0;
    float h = hash1(c + fs);
    if (h > prob) return vec3(0.0);
    h /= prob;
    vec2 ctr = c + 0.5 + (vec2(hash1(c + fs + 1.7), hash1(c + fs + 4.1)) - 0.5) * 0.4;
    float dpx = length(p - ctr) / px;
    float size = 0.75 + 1.0 * pow(hash1(c + fs + 9.2), 5.0);
    float b = 0.30 + 1.5 * pow(hash1(c + fs + 2.2), 3.0);
    b *= 0.9 + 0.1 * sin(time * (1.1 + 2.0 * h) + h * 40.0);
    float core = 1.0 - smoothstep(size * 0.45, size, dpx);
    float halo = exp(-dpx * dpx / (size * size * 2.5)) * 0.10;
    return starTint(h) * b * (core + halo);
}

// a sphere in the sky seen from the arena: direction P, angular radius R -> coverage (a) and its normal
vec4 skySphere(vec3 d, vec3 P, float R, out vec3 nrm) {
    float c = dot(d, P);
    float sinR = sin(R);
    vec3 x = d - P * c;
    float r2 = dot(x, x) / (sinR * sinR);
    nrm = vec3(0.0, 0.0, 1.0);
    if (c < 0.0 || r2 >= 1.0) return vec4(0.0);
    vec3 T1 = normalize(cross(P, vec3(0.0, 0.0, 1.0)));
    vec3 T2 = cross(T1, P);
    float u = dot(x, T1) / sinR, v = dot(x, T2) / sinR;
    nrm = normalize(T1 * u + T2 * v - P * sqrt(max(0.0, 1.0 - r2)));
    float edge = 1.0 - smoothstep(0.985, 1.0, r2);
    return vec4(1.0, 1.0, 1.0, edge);
}

// Everything in the space sky that does not move (Milky Way, nebula, gas giant + ring, moon, the planet below),
// baked once into a cube map (RSVRenderer._bake_space_sky); `open` = how much of the star field shows through.
vec3 spaceStatic(vec3 d, out float open) {
    open = 1.0;
    vec3 c = vec3(0.003, 0.004, 0.010);
    // Milky Way: a band along a great circle, dusty and mottled, with darker lanes
    vec3 N = normalize(vec3(0.35, -0.45, 0.82));
    float bd = dot(d, N);
    // tri-planar noise on the view direction: seamless and never stretched into streaks
    vec3 tw3 = pow(abs(d), vec3(4.0)); tw3 /= (tw3.x + tw3.y + tw3.z);
    float dust = fbm3(d.yz * 7.0 + 4.0) * tw3.x + fbm3(d.xz * 7.0 + 11.0) * tw3.y + fbm3(d.xy * 7.0 + 17.0) * tw3.z;
    vec2 bq = d.xy * 5.0 + d.z * 3.0;
    float band = exp(-bd * bd / 0.035) * (0.45 + 0.9 * dust);
    c += band * mix(vec3(0.035, 0.04, 0.07), vec3(0.11, 0.11, 0.14), smoothstep(0.35, 0.9, dust)) * 0.8;
    float lanes = fbm3(d.yz * 22.0) * tw3.x + fbm3(d.xz * 22.0 + 5.0) * tw3.y + fbm3(d.xy * 22.0 + 9.0) * tw3.z;
    c -= band * 0.03 * smoothstep(0.55, 0.8, lanes);
    // nebula: soft magenta / teal clouds in one region of the sky
    vec3 NB = normalize(vec3(-0.6, -0.55, 0.35));
    float nd = max(dot(d, NB), 0.0);
    float nm = fbm3(d.xy * 5.0 + d.z * 3.0) * fbm3(d.yz * 3.0 + 9.0);
    c += (vec3(0.28, 0.06, 0.26) * nm + vec3(0.03, 0.14, 0.20) * (1.0 - nm) * 0.4) * pow(nd, 7.0) * 0.8;
    c = max(c, vec3(0.0));

    // ringed gas giant
    vec3 GP = normalize(vec3(-0.55, 0.62, 0.42));
    float GR = 0.21;
    vec3 GA = normalize(vec3(0.25, -0.35, 1.0));            // spin axis = ring normal
    vec3 gn;
    vec4 g = skySphere(d, GP, GR, gn);
    // ring: a plane through the planet centre (planet at distance 1), 1.35..2.35 planet radii
    float dd = dot(d, GA);
    float tR = abs(dd) > 1e-4 ? dot(GP, GA) / dd : -1.0;
    float rr = length(d * tR - GP) / sin(GR);
    float ringA = 0.0; vec3 ringC = vec3(0.0);
    if (tR > 0.0 && rr > 1.35 && rr < 2.35) {
        float rb = fract(rr * 7.3);
        ringA = (0.35 + 0.5 * vnoise(vec2(rr * 40.0, 1.0))) * smoothstep(1.35, 1.45, rr) * (1.0 - smoothstep(2.2, 2.35, rr));
        ringA *= 1.0 - 0.75 * step(1.82, rr) * step(rr, 1.9);                                   // the gap
        ringC = mix(vec3(0.75, 0.66, 0.52), vec3(0.55, 0.50, 0.46), rb) * (0.25 + 0.9 * abs(dot(GA, SUN_DIR)));
    }
    bool ringFront = tR < dot(d, GP);                       // the ring crossing is nearer than the planet centre
    if (!ringFront) c = mix(c, ringC, ringA);
    open *= (1.0 - ringA) * (1.0 - g.a);
    if (g.a > 0.0) {
        float lat = dot(gn, GA);
        float bands = fbm3(vec2(lat * 9.0, lat * 2.0 + fbm3(vec2(lat * 30.0, dot(gn, cross(GA, GP)) * 3.0)) * 0.6));
        vec3 alb = mix(vec3(0.62, 0.45, 0.30), vec3(0.85, 0.74, 0.58), bands);
        alb = mix(alb, vec3(0.55, 0.28, 0.20), smoothstep(0.62, 0.75, fbm3(vec2(lat * 22.0, 3.0))) * 0.5);
        float lit = max(dot(gn, SUN_DIR), 0.0);
        vec3 pc = alb * (0.02 + 1.1 * lit);
        pc += vec3(0.35, 0.45, 0.7) * pow(1.0 - max(dot(gn, -d), 0.0), 3.0) * (0.15 + 0.6 * lit);   // atmosphere rim
        c = mix(c, pc, g.a);
    }
    if (ringFront) c = mix(c, ringC, ringA);
    // two moons
    vec3 mn;
    vec4 m1 = skySphere(d, normalize(vec3(0.72, 0.30, 0.38)), 0.05, mn);
    open *= 1.0 - m1.a;
    if (m1.a > 0.0) {
        float cr = fbm3(mn.xy * 9.0 + mn.z * 4.0);
        c = mix(c, vec3(0.55, 0.54, 0.52) * (0.7 + 0.5 * cr) * (0.02 + max(dot(mn, SUN_DIR), 0.0)), m1.a);
    }
    // the planet far below (the arena is in orbit): ocean, continents, clouds, a blue limb
    if (d.z < -0.28) {
        float k = (-d.z - 0.28) / 0.72;
        vec2 pq = d.xy / max(-d.z, 0.05) * 1.6 + vec2(time * 0.004, 0.0);
        float land = smoothstep(0.52, 0.58, fbm3(pq * 1.3 + 20.0));
        float cloud = smoothstep(0.5, 0.8, fbm3(pq * 2.6 + vec2(time * 0.01, 3.0)));
        vec3 surf = mix(vec3(0.02, 0.07, 0.20), mix(vec3(0.10, 0.20, 0.07), vec3(0.35, 0.30, 0.18), fbm3(pq * 5.0)), land);
        surf = mix(surf, vec3(0.85, 0.88, 0.92), cloud * 0.8);
        vec3 pn = normalize(vec3(d.xy * 0.6, 1.0));
        float lit = 0.15 + 0.85 * max(dot(pn, SUN_DIR) * 0.8 + 0.2, 0.0);
        vec3 pc = mix(vec3(0.25, 0.5, 1.0) * 0.8, surf * lit, smoothstep(0.0, 0.25, k));   // atmosphere at the limb
        c = mix(c, pc, smoothstep(0.0, 0.02, k));
        open *= 1.0 - smoothstep(0.0, 0.02, k);
    }
    c += vec3(0.15, 0.35, 0.8) * exp(-pow((d.z + 0.28) / 0.03, 2.0)) * 0.5;                    // limb glow
    return c;
}

// The live space sky: the baked cube map + the stars (twinkling, pixel-sized) and the sun on top, where open.
vec3 spaceSky(vec3 d) {
    vec4 bk = texture(spaceCube, d);
    vec3 y = pow(bk.rgb, vec3(2.2));
    vec3 c = y / max(1.0 - 0.15 * y, 0.05);                     // undo to_srgb (8-bit sRGB storage: no banding)
    float bd = dot(d, normalize(vec3(0.35, -0.45, 0.82)));
    float band = exp(-bd * bd / 0.035) * 0.9;
    vec3 st = stars(d, 50.0, 0.028, 0.0) * 1.2 + stars(d, 95.0, 0.010 + 0.03 * min(band, 1.0), 17.0) * 0.5;
    float sd = max(dot(d, SUN_DIR), 0.0);
    st += vec3(1.0, 0.97, 0.9) * (smoothstep(0.99955, 0.9997, sd) * 12.0 + pow(sd, 400.0) * 1.5 + pow(sd, 24.0) * 0.08);
    return c + st * bk.a;
}

void main() {
    vec4 a = invVP * vec4(v_ndc, -1.0, 1.0);
    vec4 b = invVP * vec4(v_ndc, 1.0, 1.0);
    vec3 d = normalize(b.xyz / b.w - a.xyz / a.w);
    if (skyBake >= 0) {
        vec2 st_ = gl_FragCoord.xy / bakeN * 2.0 - 1.0;          // GL cube map face conventions
        vec3 fd = skyBake == 0 ? vec3(1.0, -st_.y, -st_.x) : skyBake == 1 ? vec3(-1.0, -st_.y, st_.x)
                : skyBake == 2 ? vec3(st_.x, 1.0, st_.y) : skyBake == 3 ? vec3(st_.x, -1.0, -st_.y)
                : skyBake == 4 ? vec3(st_.x, -st_.y, 1.0) : vec3(-st_.x, -st_.y, -1.0);
        float open;
        vec3 c = spaceStatic(normalize(fd), open);
        f_color = vec4(to_srgb(c), open);
        return;
    }
    if (mapId == 3) {
        f_color = vec4(to_srgb(spaceSky(d)), 1.0);
        return;
    }
    vec3 c = sky_color(d);
    float cover = 0.0;
    if (d.z > 0.01) {
        // clouds: a warped fbm on a plane above the arena; lit from the sun side (a brighter rim toward the
        // sun, darker bellies), a thin high layer of wisps on top
        vec2 uv = d.xy / (d.z + 0.22) * cloudShape + vec2(time * 0.008, time * 0.002);
        vec2 w = vec2(vnoise(uv * 0.8 + 11.0), vnoise(uv * 0.8 + 37.0)) - 0.5;
        vec2 q = uv + w * 0.9;
        float dens = vnoise(q) * 0.5 + vnoise(q * 2.2 + 3.1) * 0.28 + vnoise(q * 4.7 + 7.3) * 0.14 + vnoise(q * 9.3) * 0.08;
        vec2 sdir = normalize(SUN_DIR.xy + 1e-5) * 0.18;
        vec2 q2 = q + sdir;
        float dens2 = vnoise(q2) * 0.5 + vnoise(q2 * 2.2 + 3.1) * 0.28 + vnoise(q2 * 4.7 + 7.3) * 0.14;
        float cl = smoothstep(cloudA.a, cloudA.a + 0.30, dens) * smoothstep(0.01, 0.22, d.z);
        float lit = clamp(0.5 + (dens - dens2) * 3.0, 0.0, 1.0);                          // facing the sun
        float sd = max(dot(normalize(d.xy + 1e-5), normalize(SUN_DIR.xy + 1e-5)), 0.0);
        vec3 ccol = mix(cloudB.rgb, cloudA.rgb, lit * (0.55 + 0.45 * pow(sd, 2.0)));
        ccol = mix(ccol, cloudB.rgb * 0.8, smoothstep(0.2, 0.7, d.z) * 0.5);
        ccol += uSunGlow * pow(sd, 6.0) * lit * 0.35 * (1.0 - smoothstep(0.0, 0.35, d.z));  // silver lining
        c = mix(c, ccol, cl * cloudB.a);
        float wisp = smoothstep(0.62, 0.9, vnoise(vec2(uv.x * 0.6, uv.y * 3.0) + 20.0)) * smoothstep(0.1, 0.5, d.z);
        c = mix(c, mix(cloudA.rgb, cloudB.rgb, 0.5) * 1.1, wisp * 0.18 * cloudB.a);
        cover = cl * cloudB.a;
    }
    if (mapId == 1) {
        // Forbidden Temple: the low sun itself, a soft pink-gold disc with a warm halo
        float sd = max(dot(d, SUN_DIR), 0.0);
        c += vec3(1.0, 0.78, 0.62) * smoothstep(0.99935, 0.9996, sd) * 1.6 * (1.0 - cover * 0.7);
        c += vec3(1.0, 0.55, 0.55) * pow(sd, 300.0) * 0.5;
    } else if (mapId == 2 && starK > 0.0) {
        // Parc de Paris at night: the moon
        vec3 MOON = normalize(vec3(0.55, 0.45, 0.62));
        vec3 mn;
        vec4 mm = skySphere(d, MOON, 0.032, mn);
        float halo = pow(max(dot(d, MOON), 0.0), 700.0) * 0.5 + pow(max(dot(d, MOON), 0.0), 40.0) * 0.06;
        c += vec3(0.60, 0.65, 0.90) * halo * (1.0 - cover * 0.5);
        if (mm.a > 0.0) {
            float ph = 0.25 + 0.75 * max(dot(mn, normalize(-MOON + vec3(0.35, 0.0, 0.1))), 0.0);
            float mar = vnoise(mn.xz * 6.0 + 3.0) * 0.6 + vnoise(mn.yz * 13.0) * 0.4;
            c = mix(c, vec3(0.95, 0.94, 0.98) * ph * (0.8 + 0.25 * mar), mm.a * (1.0 - cover * 0.6));
        }
    }
    // the first stars in the dark part of the dusk sky
    float sm = starK * smoothstep(0.2, 0.75, d.z) * (1.0 - cover);
    if (sm > 0.0) c += stars(d, 80.0, 0.02, 3.0) * sm;
    f_color = vec4(to_srgb(c), 1.0);
}
'''

# --------------------------------------------------------------------------------------------- #
PARTICLE_VERT = '''
#version 330
uniform mat4 m_vp;
uniform float pxScale;      // viewport_height / (2 * tan(fov / 2))
in vec3 in_pos;
in vec4 in_col;
in float in_size;
out vec4 v_col;
out float v_seed;           // >= 0: flame puff (noisy, ragged edge) with this seed; < 0: plain soft dot
void main() {
    vec4 cp = m_vp * vec4(in_pos, 1.0);
    gl_Position = cp;
    gl_PointSize = clamp(abs(in_size) * pxScale / max(cp.w, 1.0), 1.0, 512.0);
    v_col = in_col;
    v_seed = in_size < 0.0 ? fract(sin(dot(in_pos.xy, vec2(12.9898, 78.233)) + in_pos.z) * 43758.5453) * 97.0 : -1.0;
}
'''

PARTICLE_FRAG = '''
#version 330
in vec4 v_col;
in float v_seed;
out vec4 f_color;
float h2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float n2(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(h2(i), h2(i + vec2(1, 0)), f.x), mix(h2(i + vec2(0, 1)), h2(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(q, q);
    if (r2 > 1.0) discard;
    float a = v_col.a * (1.0 - r2) * (1.0 - r2);
    vec3 col = v_col.rgb;
    if (v_seed >= 0.0) {
        // flame puff: ragged, lumpy edge and a brighter, yellower core
        vec2 u = q * 2.3 + v_seed;
        float n = n2(u) * 0.62 + n2(u * 2.1 + 7.0) * 0.38;
        float body = smoothstep(0.05, 0.30, (1.0 - sqrt(r2)) * 1.25 - (n - 0.5) * 1.1);
        a = v_col.a * body * (0.55 + 0.6 * n);
        col = mix(col * (0.75 + 0.5 * n), vec3(1.0, 0.93, 0.55), (1.0 - r2) * n * 0.6);
    }
    f_color = vec4(col * a, a);      // premultiplied: works for additive (ONE,ONE) and over
}
'''

# Ball trail: a camera-facing strip shaded as a round tube (v_u = -1..1 across it -> cylinder profile).
TUBE_VERT = '''
#version 330
uniform mat4 m_vp;
in vec3 in_pos;
in vec4 in_col;
in float in_u;
out vec4 v_col;
out float v_u;
void main() { v_col = in_col; v_u = in_u; gl_Position = m_vp * vec4(in_pos, 1.0); }
'''

TUBE_FRAG = '''
#version 330
in vec4 v_col;
in float v_u;
out vec4 f_color;
void main() {
    float s = sqrt(max(1.0 - v_u * v_u, 0.0));                 // cross-section: 1 at the axis, 0 at the rim
    vec3 c = v_col.rgb * (0.45 + 0.65 * s) + vec3(1.0) * pow(s, 6.0) * 0.28;   // lit core + soft highlight
    float a = v_col.a * pow(s, 1.6);                        // fades out smoothly toward the edges
    f_color = vec4(c * a, a);
}
'''

RING_VERT = '''
#version 330
uniform mat4 m_vp;
uniform vec3 center;
uniform vec3 axisU;
uniform vec3 axisV;
uniform float radius;
in vec2 in_xy;
out vec2 v_xy;
void main() {
    v_xy = in_xy;
    vec3 p = center + (axisU * in_xy.x + axisV * in_xy.y) * radius;
    gl_Position = m_vp * vec4(p, 1.0);
}
'''

RING_FRAG = '''
#version 330
uniform vec4 color;         // rgb, alpha
uniform vec3 coreCol;
uniform float width;        // ring half-thickness as a fraction of the radius
uniform vec2 arcDir;        // (0,0) = full circle, else only the arc around this direction
uniform float arcWidth;     // cosine falloff for the arc
uniform float fill;         // faint filled disc inside the ring
uniform float sparkle;      // >0: frosted disc with twinkling sparkles (flip-reset indicator)
uniform float seed;
uniform float mode;         // 0 shockwave ring, 1 crisp badge disc, 2 soft glow disc, 3 flip-reset disc
in vec2 v_xy;
out vec4 f_color;
void main() {
    float r = length(v_xy);
    if (r > 1.0) discard;
    if (mode > 3.5) {                                   // boost pad recharge ring: lit arc = progress
        float aa = max(fwidth(r), 1e-3) * 1.5;
        float band = smoothstep(1.0 - width - aa, 1.0 - width, r) * (1.0 - smoothstep(1.0 - aa, 1.0, r));
        float ang = fract(atan(v_xy.x, v_xy.y) / 6.2831853 + 1.0);      // 0 at +v, clockwise
        float lit = 1.0 - smoothstep(fill - 0.004, fill + 0.004, ang);
        float head = exp(-pow((ang - fill) / 0.02, 2.0)) * step(fill, 0.995);   // bright leading edge
        float a = color.a * band * (0.16 + 0.84 * lit + 0.6 * head);
        vec3 c = mix(color.rgb * 0.35, color.rgb, lit) + coreCol * head * 0.8;
        f_color = vec4(c * a, a);
        return;
    }
    if (mode > 2.5) {                                   // flip-reset disc (RL): clear centre, whiter
        float aa = max(fwidth(r), 1e-3) * 1.5;          // towards the edge (~r^2), crisp white rim
        float disc = 1.0 - smoothstep(1.0 - aa, 1.0, r);
        float body = 0.08 + 0.72 * r * r;
        float rim = smoothstep(1.0 - width - aa, 1.0 - width, r) * disc;
        float a = color.a * max(body * disc, rim);
        f_color = vec4(mix(color.rgb, coreCol, rim) * a, a);
        return;
    }
    if (mode > 1.5) {                                   // jump glow: flat = bright out to ~half, soft edge;
        float g = fill > 0.5 ? pow(1.0 - r, 1.6)        // fill=1: soft ball of light, hot core
                             : 1.0 - smoothstep(0.40, 1.0, r);
        float a = color.a * g;
        f_color = vec4(mix(color.rgb, coreCol, pow(1.0 - r, 3.0)) * a, a);
        return;
    }
    if (mode > 0.5) {                                   // crisp badge: sharp white rim + frosted fill
        float aa = max(fwidth(r), 1e-3) * 1.2;
        float disc = 1.0 - smoothstep(1.0 - aa, 1.0, r);
        float rim = smoothstep(1.0 - width - aa, 1.0 - width, r) * disc;
        float f = fill * (0.75 + 0.25 * r * r);
        vec2 g = floor(v_xy * 22.0);
        float h = fract(sin(dot(g, vec2(12.9898, 78.233))) * 43758.5453);
        vec2 fc = fract(v_xy * 22.0) - 0.5;
        f += step(0.86, h) * (1.0 - smoothstep(0.05, 0.18, length(fc))) * 0.6;
        float a = color.a * max(rim, f * disc);
        vec3 c = mix(color.rgb, coreCol, rim);
        f_color = vec4(c * a, a);
        return;
    }
    float d = abs(r - (1.0 - width)) / width;
    float band = exp(-d * d * 2.2);
    float core = exp(-d * d * 14.0);
    float arc = 1.0;
    if (dot(arcDir, arcDir) > 0.0) {
        float c = dot(normalize(v_xy + 1e-5), arcDir);
        arc = smoothstep(arcWidth, 1.0, c);
    }
    float inner = step(r, 1.0 - width);
    float f = fill * inner;
    if (sparkle > 0.0) {
        f *= 0.72 + 0.28 * r * r;                                   // frosted: a bit brighter toward the rim
        vec2 g = floor(v_xy * 22.0);
        float h = fract(sin(dot(g, vec2(12.9898, 78.233))) * 43758.5453);
        vec2 fc = fract(v_xy * 22.0) - 0.5;
        float dotm = step(0.86, h) * (1.0 - smoothstep(0.05, 0.18, length(fc)));
        f += dotm * (0.6 + 0.4 * sin(seed * 25.0 + h * 60.0)) * inner;
    }
    float a = color.a * max(band, f) * arc;
    vec3 c = mix(color.rgb, coreCol, core);
    f_color = vec4(c * a, a);
}
'''

# --------------------------------------------------------------------------------------------- #
LANDSCAPE_VERT = '''
#version 330
uniform mat4 m_vp;
in vec3 in_position;
in vec2 in_uv;
out vec3 v_pos;
out vec2 v_uv;
void main() {
    v_pos = in_position;
    v_uv = in_uv;
    gl_Position = m_vp * vec4(in_position, 1.0);
}
'''

LANDSCAPE_FRAG = '''
#version 330
''' + COMMON + '''
uniform vec3 camPos;
in vec3 v_pos;
in vec2 v_uv;
out vec4 f_color;

void main() {
    // Low-poly valley: flat faces (normal from screen derivatives), per-face colour jitter,
    // aerial-perspective haze toward the dusk horizon.
    vec3 n = normalize(cross(dFdx(v_pos), dFdy(v_pos)));
    vec3 V = normalize(camPos - v_pos);
    if (dot(n, V) < 0.0) n = -n;
    float mat = floor(v_uv.y + 0.5);
    float jit = v_uv.x;
    vec3 alb;
    if (mat < 0.5) {                                      // plaza stone slabs
        vec2 q = v_pos.xy / 600.0;
        vec2 gpx = max(fwidth(q), vec2(1e-4));
        vec2 l = smoothstep(vec2(0.0), 2.0 * gpx, abs(fract(q) - 0.5) - 0.485);
        float line = max(l.x, l.y) * (1.0 - smoothstep(0.1, 0.4, max(gpx.x, gpx.y)));
        alb = vec3(0.16, 0.155, 0.17) * (0.9 + 0.2 * hash1(floor(q))) * (1.0 - 0.35 * line);
    } else if (mat < 1.5) {                               // grass hills
        alb = mix(vec3(0.10, 0.20, 0.07), vec3(0.16, 0.26, 0.09), jit);
    } else if (mat < 2.5) {                               // mountain rock: banded by height, snow caps
        float hf = jit;                                   // height fraction on peaks (0 foot .. 1 summit)
        vec3 lowR = vec3(0.30, 0.26, 0.30), midR = vec3(0.40, 0.36, 0.40), hiR = vec3(0.50, 0.47, 0.52);
        alb = hf < 0.35 ? lowR : (hf < 0.62 ? midR : hiR);
        alb *= 0.9 + 0.2 * hash1(floor(v_pos.xy / 900.0));          // facet-to-facet variation
        // snow: an absolute snow line that wanders across the range (+-1400 uu) with ragged per-pixel edges and
        // tongues reaching lower on the flatter faces -- not the same white cap on every peak
        float sline = 4300.0 + 2800.0 * (vnoise(v_pos.xy / 9000.0 + 3.0) - 0.5);
        float ragged = (vnoise(v_pos.xy / 650.0 + v_pos.z / 480.0) - 0.5) * 1100.0 + (n.z - 0.6) * 1400.0;
        float snow = smoothstep(sline - 120.0, sline + 120.0, v_pos.z + ragged) * smoothstep(0.22, 0.5, n.z);
        alb = mix(alb, vec3(0.90, 0.91, 0.96), snow);
    } else if (mat < 3.5) {                               // pine foliage
        alb = mix(vec3(0.05, 0.14, 0.07), vec3(0.08, 0.20, 0.09), jit);
    } else if (mat < 4.5) {                               // trunk
        alb = vec3(0.20, 0.12, 0.07);
    } else if (mat < 5.5) {                               // deciduous canopy: autumn palette per tree
        alb = jit < 0.3 ? vec3(0.55, 0.22, 0.05) : jit < 0.55 ? vec3(0.58, 0.42, 0.07)
            : jit < 0.72 ? vec3(0.45, 0.10, 0.05) : vec3(0.13, 0.26, 0.08);
    } else if (mat < 6.5) {                               // lake: sky reflection with fresnel, gentle ripples
        vec3 wn = normalize(vec3(0.03 * sin(v_pos.x / 160.0 + v_pos.y / 230.0), 0.03 * sin(v_pos.y / 190.0 - v_pos.x / 310.0), 1.0));
        float fres = 0.15 + 0.85 * pow(1.0 - max(dot(wn, V), 0.0), 4.0);
        vec3 refl = sky_color(reflect(-V, wn));
        vec3 wc = mix(vec3(0.02, 0.05, 0.07), refl, fres);
        float dist0 = length(camPos - v_pos);
        vec3 haze0 = sky_color(vec3(normalize(v_pos - camPos).xy, 0.02));
        wc = mix(wc, haze0, smoothstep(9000.0, 48000.0, dist0) * 0.85);
        f_color = vec4(to_srgb(wc), 1.0);
        return;
    } else {                                              // boulders
        alb = mix(vec3(0.20, 0.19, 0.21), vec3(0.30, 0.28, 0.30), jit);
    }
    // slight surface texture (tri-planar value noise at two scales, filtered by distance)
    vec3 tw = abs(n) / (abs(n.x) + abs(n.y) + abs(n.z) + 1e-4);
    vec3 tp = v_pos / (mat > 2.5 && mat < 5.5 ? 25.0 : 140.0);
    float tex = (vnoise(tp.yz) * tw.x + vnoise(tp.xz + 5.0) * tw.y + vnoise(tp.xy + 9.0) * tw.z) * 0.6
              + (vnoise(tp.yz * 3.7) * tw.x + vnoise(tp.xz * 3.7 + 2.0) * tw.y + vnoise(tp.xy * 3.7 + 4.0) * tw.z) * 0.4;
    float tfade = 1.0 - smoothstep(0.5, 2.0, length(fwidth(tp)));
    alb *= mix(1.0, 0.84 + 0.32 * tex, tfade);
    float ndl = max(dot(n, SUN_DIR), 0.0);
    vec3 amb = mix(vec3(0.10, 0.09, 0.12), vec3(0.34, 0.32, 0.42), n.z * 0.5 + 0.5) * uAmb;
    vec3 c = alb * (amb * 1.15 + SUN_COL * ndl * 0.95);
    // haze: blend toward the sky colour in the view direction (mountains fade into the dusk)
    float dist = length(camPos - v_pos);
    vec3 dir = normalize(v_pos - camPos);
    float hk = smoothstep(9000.0, 48000.0, dist) * 0.85;
    if (hk > 0.0) c = mix(c, sky_color(vec3(dir.xy, max(dir.z, 0.02))), hk);
    f_color = vec4(to_srgb(c), 1.0);
}
'''

# --------------------------------------------------------------------------------------------- #
PAD_VERT = """
#version 330
uniform mat4 m_vp;
uniform mat4 m_model;
in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord_0;
out vec3 v_pos;
out vec3 v_nrm;
out vec2 v_uv;
out float v_oz;
out vec2 v_oxy;
void main() {
    vec4 wp = m_model * vec4(in_position, 1.0);
    v_pos = wp.xyz;
    v_oz = in_position.z;
    v_oxy = in_position.xy;
    v_nrm = normalize(mat3(m_model) * in_normal);
    v_uv = in_texcoord_0;
    gl_Position = m_vp * wp;
}
"""

PAD_FRAG = """
#version 330
""" + COMMON + """
uniform sampler2D Texture;
uniform vec3 camPos;
uniform float ghost;       // >0: returning-orb pass (alpha blended), value = its fade-in 0..1
uniform float orbZ;        // object-space z where the big pad's orb starts (the gold cone below is skipped)
uniform float orbCz;       // ghost pass: object-space z of the big orb's centre (smooth sphere normals); < -100 = none
uniform float flash;       // 0..1 just-respawned flash
uniform float pulse;
uniform float charge;      // empty pad: recharge progress 0..1 (-1 = not an empty-pad draw)
uniform float padR;        // pad radius in object space (the base's outer edge)
in vec3 v_pos;
in vec3 v_nrm;
in vec2 v_uv;
in float v_oz;
in vec2 v_oxy;
out vec4 f_color;
void main() {
    vec3 tex = texture(Texture, v_uv).rgb;
    float sat = max(tex.r, max(tex.g, tex.b)) - min(tex.r, min(tex.g, tex.b));
    bool glowPart = sat > 0.25;
    vec3 n = normalize(v_nrm);
    vec3 V = normalize(camPos - v_pos);
    float ndv = abs(dot(n, V));
    if (ghost > 0.0) {
        // The orb coming back (last ~1 s of the recharge, like RL): a BLURRY whitish orb that comes into focus.
        // ghost 0 -> 1 over the window: the silhouette starts fully dissolved (opacity falls off from the centre
        // to nothing at the rim; fx.pad_glows adds a soft halo spilling past it), then the falloff band narrows
        // to a crisp edge while it firms up and warms to the orb's gold, so the pop at the respawn is small.
        // Alpha blended, so it reads against the sky from the side as well as from above.
        if (!glowPart || v_oz < orbZ) discard;
        // the orb mesh is low-poly: take the SPHERE's normal (from the orb centre) so the soft rim is smooth
        if (orbCz > -100.0) ndv = abs(dot(normalize(vec3(v_oxy, v_oz - orbCz)), V));
        // blurry and white-grey the whole time: fully soft silhouette, only its opacity fades in at the start
        float body = smoothstep(0.0, 0.95, ndv);
        vec3 c = vec3(0.80, 0.82, 0.86) * (0.9 + 0.2 * ndv);
        float a = 0.62 * body * smoothstep(0.0, 0.25, ghost);
        f_color = vec4(to_srgb(c), a);
        return;
    }
    vec3 c;
    if (glowPart) {
        // shiny gold: darker underneath, a sharp sun glint, the sky reflected on top, a hot rim
        vec3 R = reflect(-V, n);
        vec3 body = mix(vec3(0.55, 0.19, 0.02), vec3(1.0, 0.56, 0.08), smoothstep(0.0, 1.0, n.z * 0.5 + 0.5));
        c = body * (1.0 + 0.12 * pulse);
        c += vec3(1.0, 0.85, 0.55) * pow(max(dot(R, SUN_DIR), 0.0), 28.0) * 2.4;
        c += vec3(1.0, 0.92, 0.75) * pow(max(R.z, 0.0), 8.0) * 0.55;
        c += vec3(1.0, 0.70, 0.30) * pow(1.0 - ndv, 3.0) * 1.1;
        c += vec3(1.0, 0.85, 0.6) * 0.45 * flash * flash;            // soft settle after the respawn
    } else {
        float ndl = max(dot(n, SUN_DIR), 0.0);
        vec3 amb = mix(vec3(0.08), vec3(0.35, 0.34, 0.38), n.z * 0.5 + 0.5) * uAmb;
        c = tex * 0.7 * (amb * 1.3 + SUN_COL * ndl * 0.8);
        if (charge >= 0.0) {
            // Empty pad (RL): the base stays black for the first half of the recharge, then turns white, the
            // white spreading from the outer edge in to the centre (soft gradient) until the orb is back.
            float p = clamp((charge - 0.5) / 0.5, 0.0, 1.0);
            float rn = length(v_oxy) / padR;
            float front = 1.05 - 1.1 * p;                      // edge -> centre, reaching it just before the end
            float w = smoothstep(front - 0.30, front + 0.08, rn) * (0.30 + 0.70 * p) * smoothstep(0.0, 0.10, p);
            w *= smoothstep(0.2, 0.6, n.z);                    // the top surfaces only
            c = mix(c * 0.25, vec3(0.93, 0.94, 0.96) * (0.75 + 0.35 * ndl), w);
        }
    }
    f_color = vec4(to_srgb(c), 1.0);
}
"""

# --------------------------------------------------------------------------------------------- #
# Boost meter: analytic gauge on a screen quad (smooth at any size, one draw) + font-atlas digits.
GAUGE_VERT = """
#version 330
uniform mat4 m_vp;
in vec2 in_pos;       // screen px
in vec2 in_q;         // gauge-local coords, radius 1 = outer edge
out vec2 v_q;
void main() { v_q = in_q; gl_Position = m_vp * vec4(in_pos, 0.0, 1.0); }
"""

GAUGE_FRAG = """
#version 330
uniform float fill;        // 0..1 boost
uniform vec3 fillCol;
uniform vec3 fillCol2;
uniform vec3 trackCol;
uniform float startDeg;
uniform float spanDeg;
in vec2 v_q;
out vec4 f_color;
void main() {
    float r = length(v_q);
    float aa = fwidth(r) * 1.2;
    // angle: 0 = up, clockwise (screen y points down in the HUD ortho)
    float ang = degrees(atan(v_q.x, -v_q.y));
    float rel = mod(ang - startDeg + 720.0, 360.0);
    float inArc = step(rel, spanDeg);
    float t = rel / spanDeg;
    // One solid dark dial plate behind everything (so an EMPTY meter still reads as a clean dial,
    // not a translucent smear over the grass), a thin team-tinted bezel, recessed empty slots.
    float plate = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, r);
    vec3 pcol = mix(vec3(0.035, 0.037, 0.045), trackCol * 0.35, smoothstep(0.62, 1.0, r));
    float bezel = exp(-pow((r - 0.995) / 0.012, 2.0));
    vec4 c = vec4(pcol + trackCol * 1.6 * bezel, 1.0) * plate * 0.92;
    // centre disc (slightly darker, soft inner edge)
    float disc = 1.0 - smoothstep(0.64 - aa, 0.64 + aa, r);
    c.rgb = mix(c.rgb, vec3(0.012, 0.012, 0.018) * 0.92, disc);
    // ring band: segmented slots, filled ones glow in the team gradient
    float band = smoothstep(0.69 - aa, 0.69 + aa, r) * (1.0 - smoothstep(0.95 - aa, 0.95 + aa, r));
    float seg = smoothstep(0.0, 0.004 / spanDeg * 360.0, abs(fract(t * 20.0) - 0.5) - 0.46);  // tick gaps
    float filled = inArc * step(t, fill);
    vec3 fc = mix(fillCol2, fillCol, smoothstep(0.0, 1.0, t)) * (0.85 + 0.35 * smoothstep(0.69, 0.95, r));
    vec3 slot = trackCol * 0.9 + vec3(0.03);
    float slotMask = band * inArc * (1.0 - seg);
    c.rgb = mix(c.rgb, slot * 0.92, slotMask * (1.0 - filled));
    c.rgb = mix(c.rgb, fc, slotMask * filled);
    c.a = max(c.a, slotMask * filled);
    // faint halo hugging the FILLED part of the band on both sides (inner + outer edge, soft ends),
    // kept inside the dial plate so nothing spills out of the gauge
    float dBand = max(0.69 - r, r - 0.95);                          // <0 inside the band
    float tf = rel / spanDeg;
    float along = inArc * (1.0 - smoothstep(fill - 0.015, fill + 0.02, tf)) * step(0.001, fill);
    float halo = exp(-pow(max(dBand, 0.0) / 0.025, 2.0)) * step(0.0, dBand) * along * 0.22 * plate;
    c.rgb += fc * halo;
    f_color = vec4(c.rgb, c.a);        // premultiplied-ish; blended ONE, ONE_MINUS_SRC_ALPHA
}
"""

TEXT_VERT = """
#version 330
uniform mat4 m_vp;
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = m_vp * vec4(in_pos, 0.0, 1.0); }
"""

TEXT_FRAG = """
#version 330
uniform sampler2D Tex;
uniform vec4 color;
in vec2 v_uv;
out vec4 f_color;
void main() {
    float a = texture(Tex, v_uv).r * color.a;
    f_color = vec4(color.rgb * a, a);
}
"""


# --------------------------------------------------------------------------------------------- #
# Map scenery (maps.py): one static non-indexed mesh per map, every vertex = pos(3) + colour(3) + (emission,
# kind). Flat faces (normal from screen derivatives). kind: 0 matte, 1 water, 2 glowing (lanterns, lamps),
# 3 building facade with lit windows, 4 iron lattice (see-through, golden lights), 5 metal, 6 foliage,
# 7 neon strip, 9 floating lantern (bobs; emission channel = its phase).
SCENE_VERT = '''
#version 330
uniform mat4 m_vp;
uniform float time;
in vec3 in_position;
in vec3 in_col;
in vec2 in_ek;
out vec3 v_pos;
out vec3 v_col;
flat out vec2 v_ek;
void main() {
    vec3 p = in_position;
    if (abs(in_ek.y - 9.0) < 0.5 || abs(in_ek.y - 14.0) < 0.5) {
        float ph = in_ek.x;
        p.z += 90.0 * sin(time * 0.45 + ph * 6.2832);
        p.xy += 40.0 * vec2(sin(time * 0.3 + ph * 11.0), cos(time * 0.27 + ph * 7.0));
    }
    v_pos = p;
    v_col = in_col;
    v_ek = in_ek;
    gl_Position = m_vp * vec4(p, 1.0);
}
'''

SCENE_FRAG = '''
#version 330
''' + COMMON + '''
uniform vec3 camPos;
uniform float time;
uniform vec3 haze;          // near, far, strength
uniform float uNight;       // 1 = evening / night (lit windows, the tower's lights), 0 = daytime
in vec3 v_pos;
in vec3 v_col;
flat in vec2 v_ek;
out vec4 f_color;

void main() {
    vec3 n = normalize(cross(dFdx(v_pos), dFdy(v_pos)));
    vec3 V = normalize(camPos - v_pos);
    if (dot(n, V) < 0.0) n = -n;
    float kind = floor(v_ek.y + 0.5);
    float em = v_ek.x;
    vec3 alb = v_col;
    vec3 emis = vec3(0.0);
    float spec = 0.0;
    float dist = length(camPos - v_pos);
    vec3 dir = -V;
    float hk = smoothstep(haze.x, haze.y, dist) * haze.z;
    vec3 hz = hk > 0.0 ? sky_color(vec3(dir.xy, max(dir.z, 0.02))) : vec3(0.0);
    // face coordinates for the patterns: horizontal position along the wall + height
    vec2 fq = abs(n.z) > 0.8 ? v_pos.xy : (abs(n.x) > abs(n.y) ? vec2(v_pos.y, v_pos.z) : vec2(v_pos.x, v_pos.z));
    if (kind == 1.0) {                                      // water: sky reflection with fresnel, soft ripples
        vec3 wn = normalize(vec3(0.03 * sin(v_pos.x / 160.0 + v_pos.y / 230.0 + time * 0.6),
                                 0.03 * sin(v_pos.y / 190.0 - v_pos.x / 310.0 + time * 0.5), 1.0));
        float fres = 0.15 + 0.85 * pow(1.0 - max(dot(wn, V), 0.0), 4.0);
        vec3 wc = mix(alb * 0.3, sky_color(reflect(-V, wn)), fres);
        wc += SUN_COL * pow(max(dot(reflect(-V, wn), SUN_DIR), 0.0), 200.0) * 2.0;
        f_color = vec4(to_srgb(mix(wc, hz, hk)), 1.0);
        return;
    } else if (kind == 16.0) {                             // glowing paper screen behind a dark wooden lattice
        vec2 lq = fq / 55.0;
        float fwl = max(length(fwidth(lq)), 1e-4);
        vec2 lf = abs(fract(lq) - 0.5);
        float bars = (1.0 - smoothstep(0.38, 0.38 + fwl, max(lf.x, lf.y))) ;
        float det = 1.0 - smoothstep(0.3, 0.8, fwl);
        vec3 e = alb * em * mix(1.0, 0.25 + 0.75 * bars, det);
        f_color = vec4(to_srgb(mix(e, hz, hk * 0.35)), 1.0);
        return;
    } else if (kind == 2.0 || kind == 9.0) {                // lanterns / lamps: glow, a slow flicker each
        float ph = kind == 9.0 ? em * 37.0 : hash1(floor(v_pos.xy / 60.0));
        float fl = 0.95 + 0.05 * sin(time * (1.0 + 1.5 * fract(ph * 7.0)) + ph * 40.0);
        vec3 e = alb * (kind == 9.0 ? 2.2 : em) * fl;
        if (uNight < 0.5 && kind == 2.0 && em < 1.95) {          // daytime: lamps / lanterns are unlit bulbs
            vec3 cc = alb * 0.35 * (0.6 + 0.4 * max(dot(n, SUN_DIR), 0.0)) + sky_color(reflect(-V, n)) * 0.15;
            f_color = vec4(to_srgb(mix(cc, hz, hk)), 1.0);
            return;
        }
        f_color = vec4(to_srgb(mix(e, hz, hk * 0.35)), 1.0);
        return;
    } else if (kind == 3.0 && abs(n.z) < 0.35) {            // facades: a grid of windows, some lit
        vec2 wg = vec2(fq.x / 260.0, (fq.y - 60.0) / 330.0);
        vec2 wi = floor(wg), wf = fract(wg);
        float fw_ = max(length(fwidth(wg)), 1e-4);
        float win = step(0.28, wf.x) * step(wf.x, 0.72) * step(0.22, wf.y) * step(wf.y, 0.80) * step(0.0, fq.y - 60.0);
        win *= 1.0 - smoothstep(0.25, 0.6, fw_);            // far away: the average colour, no shimmer
        float lit = step(0.52, hash1(wi + floor(v_pos.xy / 3000.0) * 7.1));
        vec3 wl = mix(vec3(1.0, 0.72, 0.40), vec3(1.0, 0.86, 0.62), hash1(wi * 1.3));
        vec3 glass = mix(vec3(0.05, 0.06, 0.08), sky_color(reflect(-V, n)) * 0.55 + vec3(0.03), 1.0 - uNight);
        alb = mix(alb, glass, win);
        emis += (wl * win * lit * 1.1 + wl * lit * 0.06 * smoothstep(0.25, 0.6, fw_)) * uNight;
        // a balcony line every other floor
        alb *= 1.0 - 0.25 * step(0.9, fract((fq.y - 60.0) / 660.0)) * step(0.0, fq.y - 60.0);
    } else if (kind == 4.0) {
        // Eiffel lattice panel: v_col.xy = this panel's (u across -1..1, v up 0..1). Its borders and the X bracing
        // light up blue at a constant ~1.5 px (crisp at any distance), warm lamps glow inside the dark iron.
        vec2 uv = v_col.xy;
        vec2 fw = max(fwidth(uv), vec2(1e-5));
        if (uNight < 0.5) {
            // daytime: puddled-iron members (panel edges + X bracing, ~5% of a panel, >= 1.2 px) in bronze, the
            // panels between them a fine open lattice (see-through up close, half-transparent far away)
            float du_ = min(1.0 - abs(uv.x), min(uv.y, 1.0 - uv.y) * 2.0);
            float xs_ = abs(abs(uv.x) - abs(2.0 * uv.y - 1.0));
            float wmem = max(0.05, 1.2 * max(fw.x, fw.y));
            float member = max(1.0 - smoothstep(wmem, wmem * 1.4, du_), 1.0 - smoothstep(wmem * 0.7, wmem, xs_));
            vec2 lq2 = fq / 150.0;
            float fwl2 = max(length(fwidth(lq2)), 1e-4);
            vec2 dg2 = abs(fract(vec2(lq2.x + lq2.y, lq2.x - lq2.y) * 0.5) - 0.5);
            float fine = 1.0 - smoothstep(0.07, 0.07 + fwl2, min(dg2.x, dg2.y));
            if (member < 0.5) {
                if (fwl2 < 0.35) { if (fine < 0.5) discard; }
                else if (((int(gl_FragCoord.x) + int(gl_FragCoord.y)) & 1) == 0) discard;
            }
            alb = vec3(0.34, 0.25, 0.18) * (0.85 + 0.25 * vnoise(fq / 90.0));
            spec = 0.1;
        } else {
        float du = (1.0 - abs(uv.x)) / fw.x;
        float dv = min(uv.y, 1.0 - uv.y) / fw.y;
        float xs = abs(abs(uv.x) - abs(2.0 * uv.y - 1.0));
        float dx = xs / max(length(vec2(fw.x, 2.0 * fw.y)), 1e-5);
        float lineB = 1.0 - smoothstep(0.5, 1.3, min(du, dv));
        float lineX = 1.0 - smoothstep(0.35, 1.0, dx);
        alb = vec3(0.035, 0.035, 0.05);
        float warm = 0.10 + 0.08 * vnoise(v_pos.xy / 260.0 + v_pos.z / 210.0);
        // fine inner lattice (only up close): a small diagonal grid, a little brighter where the lamps are
        vec2 lq = fq / 150.0;
        float fwl = max(length(fwidth(lq)), 1e-4);
        vec2 dg = abs(fract(vec2(lq.x + lq.y, lq.x - lq.y) * 0.5) - 0.5);
        float bar = (1.0 - smoothstep(0.06, 0.06 + fwl, min(dg.x, dg.y))) * (1.0 - smoothstep(0.3, 0.8, fwl));
        float spark = step(0.985, hash1(floor(uv * 6.0) + floor(fq / 900.0) * 3.1 + floor(time * 1.2))) * (1.0 - smoothstep(0.2, 0.6, fwl));
        emis += vec3(1.0, 0.72, 0.38) * warm + vec3(0.25, 0.35, 0.6) * 0.10 * bar;
        emis += vec3(0.10, 0.34, 1.0) * 1.25 * max(lineB, lineX * 0.75) + vec3(0.8, 0.9, 1.0) * 1.6 * spark;
        }
    } else if (kind == 15.0) {
        // spaceship hull: plates with dark seams, some plates carry a thin strip of cool light
        vec2 hq = vec2(fq.x / 620.0, fq.y / 240.0);
        float fwh = max(length(fwidth(hq)), 1e-4);
        vec2 hf = fract(hq);
        float seam = 1.0 - smoothstep(0.0, 0.03 + fwh, min(min(hf.x, 1.0 - hf.x) * 2.6, min(hf.y, 1.0 - hf.y)));
        float h_ = hash1(floor(hq) + 3.7);
        alb *= (0.82 + 0.3 * h_) * (1.0 - 0.45 * seam * (1.0 - smoothstep(0.3, 0.8, fwh)));
        float strip = step(abs(n.z) > 0.8 ? 0.985 : 0.92, h_)                         // a strip on a few plates
                    * (1.0 - smoothstep(0.03, 0.03 + fwh, abs(hf.y - 0.5))) * step(0.12, hf.x) * step(hf.x, 0.88);
        emis += vec3(0.65, 0.88, 1.0) * strip * 1.3 * (1.0 - smoothstep(0.35, 1.0, fwh));
        emis += vec3(0.65, 0.88, 1.0) * 0.06 * step(0.55, h_) * smoothstep(0.35, 1.0, fwh);    // far: a faint average
        spec = 0.3;
    } else if (kind == 12.0) {
        // glazed barrel tiles: rounded columns running down the slope, a dark gutter under every row, the crowns
        // catching the sky -- still readable from across the arena (averages to a slightly darker roof far away)
        vec2 rq = vec2(fq.x / 70.0, v_pos.z / 55.0);
        float fwr = max(length(fwidth(rq)), 1e-4);
        float det = 1.0 - smoothstep(0.5, 1.2, fwr);
        float ridge = abs(fract(rq.x) - 0.5) * 2.0;
        float barrel = 1.0 - ridge * ridge;
        float lip = smoothstep(0.0, 0.2, fract(rq.y));
        alb *= mix(0.82, mix(0.45, 1.25, barrel * lip), det);
        emis += sky_color(reflect(-V, n)) * 0.12 * barrel * lip * det;
        spec = 0.4;
    } else if (kind == 13.0) {
        // temple walls: red lacquered timber -- dark posts every 380 uu and rails every 300 uu; a golden lattice
        // window only in some bays (about one in five), dimly lit
        vec2 wq = vec2(fq.x / 380.0, v_pos.z / 300.0);
        vec2 fwq = max(fwidth(wq), vec2(1e-4));
        float det = 1.0 - smoothstep(0.25, 0.7, max(fwq.x, fwq.y));
        vec2 fr = fract(wq);
        float post = 1.0 - smoothstep(0.05, 0.05 + fwq.x, min(fr.x, 1.0 - fr.x));
        float rail = 1.0 - smoothstep(0.04, 0.04 + fwq.y, min(fr.y, 1.0 - fr.y));
        float frame = max(post, rail);
        float bay = step(0.80, hash1(floor(wq) + floor(v_pos.xy / 5000.0) * 3.7));
        float win = bay * step(0.2, fr.x) * step(fr.x, 0.8) * step(0.45, fr.y) * step(fr.y, 0.82);
        vec2 lat = abs(fract(fq / vec2(40.0, 40.0)) - 0.5);
        float latt = win * (1.0 - smoothstep(0.10, 0.10 + fwq.x * 9.0, min(lat.x, lat.y)));
        alb = mix(alb, alb * 0.45, frame * det);
        alb = mix(alb, vec3(0.30, 0.18, 0.08), win * (1.0 - frame) * det * 0.6);
        emis += vec3(1.0, 0.62, 0.28) * (win * 0.18 + latt * 0.30) * (1.0 - frame) * det;
    }
    if (kind == 14.0) em = 0.0;                           // (the emission channel carried the bob phase)
    if (kind == 6.0) alb *= 0.85 + 0.3 * hash1(floor(v_pos.xy / 120.0 + v_pos.z / 90.0));
    if (kind == 8.0) {
        // karst limestone: vertical streaks; vegetation patches from a noise field (NOT per face -- that showed
        // the triangles), more of it higher up and on gentler slopes. em = height fraction (per vertex).
        float zf = em;
        float st_ = vnoise(vec2((v_pos.x + v_pos.y) / 170.0, v_pos.z / 2200.0));
        alb = v_col * (0.80 + 0.35 * st_);
        float nse = vnoise(v_pos.xy / 900.0 + v_pos.z / 650.0) * 0.6 + vnoise(v_pos.yx / 310.0 + v_pos.z / 240.0) * 0.4;
        float veg = smoothstep(0.50, 0.62, nse + zf * 0.42 + n.z * 0.18 - 0.2);
        vec3 green = mix(vec3(0.11, 0.21, 0.12), vec3(0.18, 0.30, 0.15), vnoise(v_pos.xy / 140.0 + v_pos.z / 140.0));
        alb = mix(alb, green, veg);
        em = 0.0;
    } else if (kind == 10.0) {
        // asteroid rock: tri-planar grain + darker crater-like spots
        vec3 tw = abs(n) / (abs(n.x) + abs(n.y) + abs(n.z) + 1e-4);
        vec3 tp = v_pos / 160.0;
        float tx = vnoise(tp.yz) * tw.x + vnoise(tp.xz + 3.0) * tw.y + vnoise(tp.xy + 7.0) * tw.z;
        vec3 tp2 = v_pos / 520.0;
        float cr = vnoise(tp2.yz) * tw.x + vnoise(tp2.xz + 5.0) * tw.y + vnoise(tp2.xy + 9.0) * tw.z;
        alb *= (0.78 + 0.4 * tx) * (1.0 - 0.45 * smoothstep(0.62, 0.72, cr)) * (1.0 + 0.25 * smoothstep(0.72, 0.8, cr));
        spec = 0.08;
    } else if (kind == 11.0) {
        // graffiti: dark concrete, filled spray shapes in a few loud colours with black outlines, white
        // highlight strokes, the odd drip
        vec2 gq = fq / 320.0;
        float fwg = max(length(fwidth(gq)), 1e-4);
        float n1 = vnoise(gq * 1.3) * 0.65 + vnoise(gq * 2.9 + 5.0) * 0.35;
        float n2 = vnoise(gq * 0.6 + 13.0);
        vec3 gc = n2 < 0.3 ? vec3(0.98, 0.20, 0.55) : (n2 < 0.5 ? vec3(0.15, 0.80, 0.98)
                : (n2 < 0.7 ? vec3(1.0, 0.85, 0.10) : vec3(0.55, 0.95, 0.25)));
        float fillg = smoothstep(0.555 - fwg, 0.555 + fwg, n1);
        float outl = 1.0 - smoothstep(0.012, 0.012 + fwg * 2.0, abs(n1 - 0.555));
        float hi = (1.0 - smoothstep(0.006, 0.006 + fwg * 2.0, abs(n1 - 0.64))) * step(0.64, n1 + 0.01);
        float drip = step(0.93, hash1(vec2(floor(gq.x * 6.0), 1.0))) * step(fract(gq.x * 6.0), 0.12)
                   * step(0.40, n1) * (1.0 - smoothstep(0.0, 0.8, gq.y - floor(gq.y)));
        alb = vec3(0.17, 0.17, 0.19) * (0.9 + 0.2 * vnoise(gq * 8.0));
        alb = mix(alb, gc, max(fillg, drip * 0.8));
        alb = mix(alb, vec3(0.03), outl);
        alb = mix(alb, vec3(0.95), hi);
        emis += alb * 0.30 * fillg;
    }
    float ndl = max(dot(n, SUN_DIR), 0.0);
    vec3 amb = mix(vec3(0.10, 0.09, 0.12), vec3(0.34, 0.32, 0.42), n.z * 0.5 + 0.5) * uAmb;
    vec3 c = alb * (amb * 1.15 + SUN_COL * ndl * 0.95);
    c += SUN_COL * spec * pow(max(dot(n, normalize(SUN_DIR + V)), 0.0), 40.0);
    c += alb * em * (kind == 0.0 || kind == 6.0 ? 1.0 : 0.0);          // self-lit tint (e.g. lit paper walls)
    c = mix(c, hz, hk);
    c += emis * (1.0 - hk * 0.6);
    f_color = vec4(to_srgb(c), 1.0);
}
'''

# The crowd: every egg is ONE camera-facing quad (turning about the vertical only), shaded as an egg in the
# fragment shader (silhouette, normal, gloss) -- 2 triangles per fan instead of a 56-triangle mesh, which dropped
# frames with 9k fans. Per instance: position, colour, (phase, scale). `cheer` (0..1) makes them jump on their
# seats after a goal or a save. Drawn with alpha-to-coverage so the silhouettes are antialiased under MSAA.
CROWD_VERT = '''
#version 330
uniform mat4 m_vp;
uniform vec3 camPos;
uniform float time;
uniform vec2 cheerAge;       // seconds since the last goal / save this crowd cheers for: (blue fans, orange fans)
in vec2 in_corner;           // x -1..1, y 0..1
in vec3 i_pos;
in vec3 i_col;
in vec2 i_ps;
out vec2 v_uv;
out vec3 v_pos;
flat out vec3 v_col;
flat out vec3 v_R;
flat out vec3 v_F;
void main() {
    // i_ps.x = crowd (integer part: 0 neutral, 1 blue fans, 2 orange fans) + phase (fraction)
    float team = floor(i_ps.x), ph = fract(i_ps.x), sc = i_ps.y;
    float age = team < 0.5 ? min(cheerAge.x, cheerAge.y) : (team < 1.5 ? cheerAge.x : cheerAge.y);
    // after a goal / save every fan joins after its own delay and stops at its own time, easing in and out
    float delay = fract(ph * 5.13) * 0.7;
    float dur = 2.8 + fract(ph * 9.71) * 3.5;
    float ev = smoothstep(delay, delay + 0.45, age) * (1.0 - smoothstep(dur - 1.4, dur, age));
    // and about 30% of the fans are always cheering, each drifting in and out over several seconds
    float nb = 0.5 + 0.25 * sin(time * 0.23 + ph * 61.0) + 0.25 * sin(time * 0.37 + ph * 23.0 + 1.3);
    float idle = smoothstep(0.48, 0.62, nb) * 0.65;
    float cheer = max(ev, idle);
    float jumper = step(0.12, fract(ph * 17.3));                          // a few never get up
    float f = 7.0 + 4.5 * fract(ph * 3.1);
    float hop = abs(sin(time * f + ph * 6.2832 + 2.0 * sin(time * 0.5 + ph * 30.0)));
    float j = cheer * jumper * (16.0 + 26.0 * fract(ph * 7.3)) * sc / 60.0 * hop;
    float idleb = 1.2 * sin(time * 1.3 + ph * 40.0);
    float stretch = 1.0 + 0.10 * cheer * jumper * (hop - 0.5);           // stretch in the air, squash landing
    vec3 base = i_pos + vec3(0.0, 0.0, idleb + j);
    vec3 to = camPos - base;
    vec3 F = normalize(vec3(to.xy, 0.0) + vec3(1e-4, 0.0, 0.0));
    vec3 R = vec3(-F.y, F.x, 0.0);
    vec3 p = base + R * in_corner.x * 0.40 * sc + vec3(0.0, 0.0, in_corner.y * sc * stretch);
    v_uv = in_corner;
    v_pos = p;
    v_col = i_col;
    v_R = R;
    v_F = F;
    gl_Position = m_vp * vec4(p, 1.0);
}
'''

CROWD_FRAG = '''
#version 330
''' + COMMON + '''
uniform vec3 camPos;
uniform vec3 haze;
in vec2 v_uv;
in vec3 v_pos;
flat in vec3 v_col;
flat in vec3 v_R;
flat in vec3 v_F;
out vec4 f_color;
void main() {
    // egg silhouette: an ellipse, a little wider low than high
    float y = v_uv.y;
    float halfw = 0.36 * (1.0 - 0.22 * (y - 0.45)) / 0.40;
    vec2 e = vec2(v_uv.x / halfw, (y - 0.47) / 0.53);
    float rr = dot(e, e);
    float aa = max(fwidth(rr), 1e-3);
    float cov = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, rr);
    if (cov <= 0.0) discard;
    vec3 nl = vec3(e.x, e.y, sqrt(max(0.0, 1.0 - min(rr, 1.0))));
    vec3 n = normalize(v_R * nl.x + vec3(0.0, 0.0, 1.0) * nl.y + v_F * nl.z);
    vec3 V = normalize(camPos - v_pos);
    float ndl = max(dot(n, SUN_DIR), 0.0);
    vec3 amb = mix(vec3(0.12, 0.11, 0.14), vec3(0.42, 0.40, 0.48), n.z * 0.5 + 0.5) * uAmb;
    vec3 c = v_col * (amb * 1.3 + SUN_COL * (ndl * 0.8 + 0.15));
    c += SUN_COL * 0.35 * pow(max(dot(n, normalize(SUN_DIR + V)), 0.0), 36.0);     // glossy shell
    c += v_col * 0.25 * pow(1.0 - nl.z, 3.0);                                        // soft rim
    float dist = length(camPos - v_pos);
    vec3 dir = -V;
    float hk = smoothstep(haze.x, haze.y, dist) * haze.z;
    if (hk > 0.0) c = mix(c, sky_color(vec3(dir.xy, max(dir.z, 0.02))), hk);
    f_color = vec4(to_srgb(c), cov);
}
'''


# The grass blade pattern, baked once into a repeating texture (128 x 128 cells, 16 texels per cell): individual
# strands as short, randomly oriented blades, darker at the root and bright at the tip (value = the brightest blade
# covering the texel). The arena shader reads it for its two blade layers.
BLADE_FRAG = '''
#version 330
float hash1(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
out vec4 f_color;
void main() {
    const float N = 128.0;
    vec2 qq = gl_FragCoord.xy / 16.0;
    vec2 cid = floor(qq), f = fract(qq);
    float best = 0.0;
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
        vec2 c = mod(cid + vec2(i, j), N);
        float h1 = hash1(c), h2 = hash1(c + 31.7), h3 = hash1(c + 57.1);
        vec2 root = vec2(i, j) + vec2(h1, h2);
        float ang = h3 * 6.2832;
        vec2 dir = vec2(cos(ang), sin(ang));
        float len = 0.9 + 0.8 * hash1(c + 91.3);
        vec2 d = f - root;
        float along = clamp(dot(d, dir), 0.0, len);
        float dist = length(d - dir * along);
        float w = 0.10 * (1.0 - 0.7 * along / len);
        float blade = (1.0 - smoothstep(w, w + 0.06, dist)) * (0.35 + 0.65 * along / len);
        best = max(best, blade * (0.6 + 0.4 * hash1(c + 7.7)));
    }
    f_color = vec4(best, 0.0, 0.0, 1.0);
}
'''

# The turf grain: 6, 12, 24, 48 uu value-noise octaves (weights 0.6, 1, 1, 1), periodic over 768 uu (2048 texels),
# stored as 0.5 + grain / 4.
GRAIN_FRAG = '''
#version 330
float hash1(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float pnoise(vec2 p, float L, vec2 o) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    vec2 a = mod(i, L), b = mod(i + 1.0, L);
    return mix(mix(hash1(a + o), hash1(vec2(b.x, a.y) + o), f.x), mix(hash1(vec2(a.x, b.y) + o), hash1(b + o), f.x), f.y);
}
out vec4 f_color;
void main() {
    vec2 q = gl_FragCoord.xy * (768.0 / 2048.0);
    float g = 0.0, sc = 6.0;
    for (int k = 0; k < 4; k++) {
        g += (pnoise(q / sc, 768.0 / sc, vec2(17.3 * float(k), 5.1 * float(k))) - 0.5) * (k == 0 ? 0.6 : 1.0);
        sc *= 2.0;
    }
    f_color = vec4(clamp(0.5 + g * 0.25, 0.0, 1.0), 0.0, 0.0, 1.0);
}
'''
