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
const vec3 SUN_DIR = normalize(vec3(-0.45, 0.75, 0.30));   // low evening sun
const vec3 SUN_COL = vec3(1.00, 0.88, 0.72);

float hash1(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash1(i), hash1(i + vec2(1, 0)), f.x), mix(hash1(i + vec2(0, 1)), hash1(i + vec2(1, 1)), f.x), f.y);
}

// Evening sky (used for the background AND for reflections).
vec3 sky_color(vec3 d) {
    float h = d.z;
    vec3 zen = vec3(0.035, 0.05, 0.13);
    vec3 mid = vec3(0.22, 0.15, 0.30);
    vec3 hor = vec3(0.95, 0.46, 0.20);
    vec3 c = mix(hor, mid, smoothstep(-0.02, 0.28, h));
    c = mix(c, zen, smoothstep(0.28, 0.85, h));
    float sd = max(dot(d, SUN_DIR), 0.0);
    c += vec3(1.0, 0.55, 0.25) * (pow(sd, 8.0) * 0.45 + pow(sd, 90.0) * 1.2);
    c = mix(c, vec3(0.04, 0.035, 0.05), smoothstep(0.0, -0.12, h));   // below horizon
    return c;
}

vec3 to_srgb(vec3 c) {
    c = c / (1.0 + 0.15 * c);
    return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
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
uniform vec4 casters[9];
uniform vec2 casterFwd[9];
uniform int nCasters;
uniform vec3 blueCol;
uniform vec3 orangeCol;
uniform int passMode;          // 0 = opaque parts, 1 = translucent walls + ceiling
uniform float time;
uniform float detailBias;      // 1 = smooth distant detail, 2 = "sharp" (detail kept twice as far)
uniform vec4 ballMark;         // RL ball marker: ball x, y, centre z, height factor 0..1 (< 0 = off)

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
float sdRoundBox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
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
    float fw = max(fwidth(r), 1e-3);
    float t = clamp(ballMark.w, 0.0, 1.0);
    const float RO = 91.25;                                       // the outer ring = the ball's size
    float wo = max(2.4, 0.8 * fw);
    float outer = 1.0 - smoothstep(wo - fw, wo + fw, abs(r - RO));
    float ri = RO * mix(0.80, 0.33, t);
    float hs = 0.72 * pow(1.0 - t, 0.8);                         // half angle of each arc (rad)
    float a = atan(d.y, d.x) - 0.78539816;                        // arcs centred on the diagonals
    float dl = a - floor(a / 1.5707963 + 0.5) * 1.5707963;
    float ang = a - dl + clamp(dl, -hs, hs) + 0.78539816;
    float di = length(d - ri * vec2(cos(ang), sin(ang)));
    float wi = max(mix(2.4, 3.6, t), 0.8 * fw);                   // the dots a bit fatter than the line
    float inner = 1.0 - smoothstep(wi - fw, wi + fw, di);
    // any surface facing up at all (floor, the whole floor-wall curve up to where it turns vertical), not above
    // the ball's top: a ball resting against the wall projects onto the curve higher than its centre
    float on = step(0.0, ballMark.w) * step(p.z, ballMark.z + 91.25) * step(0.03, n.z) * step(r, 120.0);
    return max(outer, inner) * on;
}
float shadowAt(vec3 p) {
    float sh = 0.0;
    for (int i = 0; i < nCasters; i++) {
        vec4 c = casters[i];
        float h = c.z - p.z;
        if (h < -20.0) continue;
        vec2 d = p.xy - c.xy;
        vec2 f = casterFwd[i];
        if (dot(f, f) > 0.0) { vec2 fr = vec2(-f.y, f.x); d = vec2(dot(d, f) / 1.55, dot(d, fr)); }
        float soft = 1.0 + h / 900.0;
        float r = c.w * soft;
        float a = 1.0 - smoothstep(r * 0.45, r, length(d));
        sh = max(sh, a * (1.0 - smoothstep(300.0, 2600.0, h)) * (0.6 / soft));
    }
    return sh;
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
            col = vec3(0.080, 0.265, 0.050);
            // blades fade over a wider band, on a footprint between the longest axis (no shimmer) and the
            // area (sharp), so there's no visible line where the strands stop
            float pxb = mix(px, sqrt(max(length(dFdx(q)) * length(dFdy(q)), 1e-6)) / max(detailBias, 1.0), 0.5);
            float fineW = 1.0 - smoothstep(0.7, 4.5, pxb);
            if (fineW > 0.0) {
                float blades = 0.0;
                for (int L = 0; L < 2; L++) {
                    float cs = L == 0 ? 3.2 : 2.1;
                    vec2 qq = (L == 0 ? q : mat2(0.8, -0.6, 0.6, 0.8) * q + 11.0) / cs;
                    vec2 cid = floor(qq), f = fract(qq);
                    float best = 0.0;
                    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
                        vec2 c = cid + vec2(i, j);
                        float h1 = hash1(c), h2 = hash1(c + 31.7), h3 = hash1(c + 57.1);
                        vec2 root = vec2(i, j) + vec2(h1, h2);
                        float ang = h3 * 6.2832;
                        vec2 dir = vec2(cos(ang), sin(ang));
                        float len = 0.9 + 0.8 * hash1(c + 91.3);
                        vec2 d = f - root;
                        float along = clamp(dot(d, dir), 0.0, len);
                        float dist = length(d - dir * along);
                        float w = 0.10 * (1.0 - 0.7 * along / len);                    // tapers to the tip
                        float blade = (1.0 - smoothstep(w, w + 0.06, dist)) * (0.35 + 0.65 * along / len);
                        best = max(best, blade * (0.6 + 0.4 * hash1(c + 7.7)));
                    }
                    blades += best * (L == 0 ? 0.6 : 0.4);
                }
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
            float grain = 0.0;
            float sc = 6.0;
            mat2 rot = mat2(0.8, -0.6, 0.6, 0.8);
            vec2 qo = q;
            for (int k = 0; k < 4; k++) {          // 6, 12, 24, 48 uu octaves, each kept until ~1.5 px
                grain += (vnoise(qo / sc) - 0.5) * smoothstep(1.5, 4.0, sc / pxa) * (k == 0 ? 0.6 : 1.0);
                sc *= 2.0; qo = rot * qo + 17.3;
            }
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
            // paint: zones sit on slightly darker turf, team colour over it, dark chevrons over solid fills
            col *= 1.0 - 0.18 * zone;
            col = mix(col, tc, fill * 0.92);
            col = mix(col, col * 0.25, dark * fill);
            col = mix(col, vec3(0.80, 0.83, 0.80), white * 0.8);
            emis += tc * fill * (1.0 - dark) * 0.12;
            spec = 0.03;
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
        alpha = 0.13 + 0.12 * fres + hl * 0.55;              // no floor-level glow band
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

    float bmk = ballMarkAt(p, n) * (ceil ? 0.0 : 1.0);
    col = mix(col, vec3(0.85), bmk * 0.85);
    emis = mix(emis, vec3(0.40), bmk);
    alpha = max(alpha, bmk * 0.9);                                // on the glass part of the curve too

    // ---- lighting ----
    float ndl = max(dot(n, SUN_DIR), 0.0);
    vec3 amb = mix(vec3(0.12, 0.11, 0.12), vec3(0.42, 0.40, 0.46), n.z * 0.5 + 0.5);
    float sh = grid ? shadowAt(p) : 0.0;
    vec3 lit = col * (amb * 1.25 + SUN_COL * ndl * 0.75 * (1.0 - sh));
    vec3 H = normalize(SUN_DIR + V);
    lit += SUN_COL * spec * pow(max(dot(n, H), 0.0), 30.0) * (1.0 - sh);
    lit += emis;
    float dist = length(camPos - p);
    lit = mix(lit, vec3(0.30, 0.20, 0.22), smoothstep(5000.0, 16000.0, dist) * 0.35);
    f_color = vec4(to_srgb(lit) * (passMode == 1 ? alpha : 1.0), alpha);   // premultiplied in pass 1
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
    vec3 amb = mix(vec3(0.07, 0.07, 0.07), vec3(0.32, 0.35, 0.42), n.z * 0.5 + 0.5);
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
    vec3 amb = mix(vec3(0.10, 0.10, 0.11), vec3(0.40, 0.41, 0.44), n.z * 0.5 + 0.5);
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

void main() {
    vec4 a = invVP * vec4(v_ndc, -1.0, 1.0);
    vec4 b = invVP * vec4(v_ndc, 1.0, 1.0);
    vec3 d = normalize(b.xyz / b.w - a.xyz / a.w);
    vec3 c = sky_color(d);
    if (d.z > 0.02) {                                     // clouds
        vec2 uv = d.xy / (d.z + 0.25) * 2.2 + vec2(time * 0.01, 0.0);
        float cl = vnoise(uv) * 0.6 + vnoise(uv * 2.3) * 0.3 + vnoise(uv * 5.1) * 0.1;
        cl = smoothstep(0.52, 0.85, cl) * smoothstep(0.02, 0.25, d.z);
        vec3 ccol = mix(vec3(0.95, 0.55, 0.35), vec3(0.35, 0.25, 0.40), smoothstep(0.1, 0.6, d.z));
        c = mix(c, ccol, cl * 0.55);
    }
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
void main() {
    vec4 cp = m_vp * vec4(in_pos, 1.0);
    gl_Position = cp;
    gl_PointSize = clamp(in_size * pxScale / max(cp.w, 1.0), 1.0, 512.0);
    v_col = in_col;
}
'''

PARTICLE_FRAG = '''
#version 330
in vec4 v_col;
out vec4 f_color;
void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(q, q);
    if (r2 > 1.0) discard;
    float a = v_col.a * (1.0 - r2) * (1.0 - r2);
    f_color = vec4(v_col.rgb * a, a);      // premultiplied: works for additive (ONE,ONE) and over
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
        float snow = step(0.66, hf) * smoothstep(0.25, 0.55, n.z);
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
    vec3 amb = mix(vec3(0.10, 0.09, 0.12), vec3(0.34, 0.32, 0.42), n.z * 0.5 + 0.5);
    vec3 c = alb * (amb * 1.15 + SUN_COL * ndl * 0.95);
    // haze: blend toward the sky colour in the view direction (mountains fade into the dusk)
    float dist = length(camPos - v_pos);
    vec3 dir = normalize(v_pos - camPos);
    vec3 haze = sky_color(vec3(dir.xy, max(dir.z, 0.02)));
    c = mix(c, haze, smoothstep(9000.0, 48000.0, dist) * 0.85);
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
        float sharp = ghost * ghost * (3.0 - 2.0 * ghost);             // eased 0..1
        float band = mix(1.0, 0.16, sharp);                            // width of the soft rim (1 = all soft)
        float body = smoothstep(0.0, band, ndv);
        vec3 c = mix(vec3(0.84, 0.88, 0.97), vec3(1.0, 0.64, 0.18), smoothstep(0.45, 1.0, sharp));
        c *= 0.95 + 0.35 * pow(1.0 - ndv, 2.0) * sharp;                // a rim only once it is in focus
        float a = mix(0.30, 0.90, sharp) * body * smoothstep(0.0, 0.2, ghost);
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
        vec3 amb = mix(vec3(0.08), vec3(0.35, 0.34, 0.38), n.z * 0.5 + 0.5);
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
