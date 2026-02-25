import glfw
import OpenGL.GL as gl
from OpenGL.GLU import gluPerspective, gluLookAt, gluNewQuadric, gluCylinder, gluDisk, gluDeleteQuadric
import numpy as np
import math
from dronedynamics import DroneDynamics
from control import INDIController

WIDTH, HEIGHT = 1280, 900


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_matrix(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Geometry primitives
# ─────────────────────────────────────────────────────────────────────────────

def draw_box(w, h, d):
    """Solid box centered at origin with per-face shading."""
    x, y, z = w / 2, h / 2, d / 2
    faces = [
        # normal,           vertices (CCW)
        (( 0,  0,  1), [(-x,-y, z),( x,-y, z),( x, y, z),(-x, y, z)]),  # front
        (( 0,  0, -1), [(-x,-y,-z),(-x, y,-z),( x, y,-z),( x,-y,-z)]),  # back
        ((-1,  0,  0), [(-x,-y,-z),(-x,-y, z),(-x, y, z),(-x, y,-z)]),  # left
        (( 1,  0,  0), [( x,-y,-z),( x, y,-z),( x, y, z),( x,-y, z)]),  # right
        (( 0,  1,  0), [(-x, y,-z),(-x, y, z),( x, y, z),( x, y,-z)]),  # top
        (( 0, -1,  0), [(-x,-y,-z),( x,-y,-z),( x,-y, z),(-x,-y, z)]),  # bottom
    ]
    gl.glBegin(gl.GL_QUADS)
    for normal, verts in faces:
        gl.glNormal3f(*normal)
        for v in verts:
            gl.glVertex3f(*v)
    gl.glEnd()


def draw_cylinder_z(radius, height, slices=20):
    q = gluNewQuadric()
    gluCylinder(q, radius, radius, height, slices, 1)
    gluDeleteQuadric(q)


def draw_disk_z(radius, slices=32):
    q = gluNewQuadric()
    gluDisk(q, 0.0, radius, slices, 1)
    gluDeleteQuadric(q)


# ─────────────────────────────────────────────────────────────────────────────
# Drone 3-D model
# ─────────────────────────────────────────────────────────────────────────────

#  Motor layout (top view):
#    2(CCW)  0(CW)       ← Motor indices match DroneDynamics
#    3(CW)   1(CCW)
#
#  Colour convention:  red = CW motors (0,3),  blue = CCW motors (1,2)

MOTOR_POS = [
    np.array([ 0.25,  0.25, 0.0]),
    np.array([-0.25,  0.25, 0.0]),
    np.array([-0.25, -0.25, 0.0]),
    np.array([ 0.25, -0.25, 0.0]),
]
MOTOR_COLORS = [
    (0.90, 0.20, 0.20),   # 0 – CW  – red
    (0.20, 0.45, 0.90),   # 1 – CCW – blue
    (0.20, 0.45, 0.90),   # 2 – CCW – blue
    (0.90, 0.20, 0.20),   # 3 – CW  – red
]
SPIN_DIR = [1, -1, 1, -1]   # +1 = CCW, -1 = CW (matches DroneDynamics)


def draw_drone_model(Omega_cmd, sim_time):
    """
    Render the drone in its own body frame (origin = CoM).
    Omega_cmd  – motor speed commands (rad/s), used for propeller animation.
    sim_time   – simulation clock for prop rotation.
    """

    # --- Central body --------------------------------------------------------
    gl.glColor3f(0.15, 0.15, 0.22)
    draw_box(0.20, 0.20, 0.07)

    # LED strip on top (white)
    gl.glColor3f(0.9, 0.9, 0.9)
    draw_box(0.12, 0.12, 0.005)

    # --- Arms ----------------------------------------------------------------
    for i, mpos in enumerate(MOTOR_POS):
        col = [c * 0.55 for c in MOTOR_COLORS[i]]
        gl.glColor3f(*col)
        # Thin box from origin to motor position
        direction = mpos / np.linalg.norm(mpos)
        mid = mpos * 0.5
        length = np.linalg.norm(mpos)
        gl.glPushMatrix()
        gl.glTranslatef(*mid)
        # Rotate arm along drone diagonal
        angle_deg = math.degrees(math.atan2(mpos[1], mpos[0]))
        gl.glRotatef(angle_deg, 0, 0, 1)
        draw_box(length, 0.040, 0.025)
        gl.glPopMatrix()

    # --- Motor pods + propellers ---------------------------------------------
    for i, mpos in enumerate(MOTOR_POS):
        col = MOTOR_COLORS[i]

        gl.glPushMatrix()
        gl.glTranslatef(*mpos)

        # Motor cylinder
        gl.glColor3f(*col)
        gl.glPushMatrix()
        gl.glTranslatef(0, 0, -0.030)
        draw_cylinder_z(0.038, 0.050, slices=18)
        # Top cap
        gl.glTranslatef(0, 0, 0.050)
        draw_disk_z(0.038)
        gl.glPopMatrix()

        # Propeller disc (spinning, semi-transparent)
        omega = float(Omega_cmd[i]) if i < len(Omega_cmd) else 300.0
        prop_angle = math.degrees(sim_time * omega * SPIN_DIR[i]) % 360.0

        gl.glPushMatrix()
        gl.glTranslatef(0, 0, 0.030)
        gl.glRotatef(prop_angle, 0, 0, 1)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST)   # always draw on top of pod

        # Two blades per propeller
        blade_r, blade_w = 0.155, 0.018
        for b_angle in (0, 90):
            gl.glPushMatrix()
            gl.glRotatef(b_angle, 0, 0, 1)
            gl.glColor4f(*col, 0.65)
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(-blade_r, -blade_w, 0)
            gl.glVertex3f( blade_r, -blade_w, 0)
            gl.glVertex3f( blade_r,  blade_w, 0)
            gl.glVertex3f(-blade_r,  blade_w, 0)
            gl.glEnd()
            gl.glPopMatrix()

        # Thin disc for motion blur feel
        gl.glColor4f(*col, 0.12)
        draw_disk_z(blade_r)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_BLEND)
        gl.glPopMatrix()  # propeller

        gl.glPopMatrix()  # motor pod


# ─────────────────────────────────────────────────────────────────────────────
# Scene elements
# ─────────────────────────────────────────────────────────────────────────────

def draw_grid(size=20, step=1):
    """Checkerboard-ish grid at z = 0."""
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glLineWidth(1.0)
    for i in range(-size, size + 1):
        alpha = 0.55 if i % 5 == 0 else 0.22
        gl.glColor4f(0.35, 0.60, 0.35, alpha)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(i * step, -size * step, 0)
        gl.glVertex3f(i * step,  size * step, 0)
        gl.glVertex3f(-size * step, i * step, 0)
        gl.glVertex3f( size * step, i * step, 0)
        gl.glEnd()
    gl.glDisable(gl.GL_BLEND)


def draw_world_axes(length=1.2):
    gl.glLineWidth(2.5)
    gl.glBegin(gl.GL_LINES)
    gl.glColor3f(1, 0.2, 0.2); gl.glVertex3f(0,0,0); gl.glVertex3f(length,0,0)
    gl.glColor3f(0.2, 1, 0.2); gl.glVertex3f(0,0,0); gl.glVertex3f(0,length,0)
    gl.glColor3f(0.2, 0.2, 1); gl.glVertex3f(0,0,0); gl.glVertex3f(0,0,length)
    gl.glEnd()
    gl.glLineWidth(1.0)


def draw_shadow(pos):
    x, y, z = pos
    z = max(z, 0.01)
    scale = max(0.05, 1.0 - z * 0.12)
    alpha = max(0.0, 0.45 - z * 0.06)
    gl.glPushMatrix()
    gl.glTranslatef(x, y, 0.003)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glColor4f(0, 0, 0, alpha)
    gl.glBegin(gl.GL_TRIANGLE_FAN)
    gl.glVertex3f(0, 0, 0)
    for k in range(33):
        a = 2 * math.pi * k / 32
        gl.glVertex3f(0.55 * scale * math.cos(a), 0.55 * scale * math.sin(a), 0)
    gl.glEnd()
    gl.glDisable(gl.GL_BLEND)
    gl.glPopMatrix()


def draw_altitude_target(z_target, drone_x, drone_y, pulse):
    """Green crosshair ring that pulses slightly."""
    gl.glPushMatrix()
    gl.glTranslatef(drone_x, drone_y, z_target)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    r = 0.30 + 0.03 * pulse
    ring_alpha = 0.7 + 0.15 * pulse

    # Outer ring
    gl.glLineWidth(2.5)
    gl.glColor4f(0.15, 1.0, 0.15, ring_alpha)
    gl.glBegin(gl.GL_LINE_LOOP)
    for k in range(64):
        a = 2 * math.pi * k / 64
        gl.glVertex3f(r * math.cos(a), r * math.sin(a), 0)
    gl.glEnd()

    # Cross gaps
    gl.glLineWidth(1.5)
    gl.glColor4f(0.15, 1.0, 0.15, ring_alpha * 0.9)
    gl.glBegin(gl.GL_LINES)
    for sign in (+1, -1):
        gl.glVertex3f(sign * (r + 0.15), 0, 0)
        gl.glVertex3f(sign * (r + 0.40), 0, 0)
        gl.glVertex3f(0, sign * (r + 0.15), 0)
        gl.glVertex3f(0, sign * (r + 0.40), 0)
    gl.glEnd()

    # Vertical dashed line to ground (draw as short segments)
    gl.glColor4f(0.15, 1.0, 0.15, 0.25)
    gl.glBegin(gl.GL_LINES)
    steps = int(z_target / 0.3)
    for k in range(steps):
        if k % 2 == 0:
            z0 = -z_target + k * 0.3
            z1 = z0 + 0.20
            gl.glVertex3f(0, 0, z0)
            gl.glVertex3f(0, 0, z1)
    gl.glEnd()

    gl.glLineWidth(1.0)
    gl.glDisable(gl.GL_BLEND)
    gl.glPopMatrix()


def draw_drone_altitude_line(pos):
    """Dotted vertical line from drone straight down to ground."""
    x, y, z = pos
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glColor4f(1.0, 1.0, 0.3, 0.40)
    gl.glLineWidth(1.2)
    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(x, y, 0)
    gl.glVertex3f(x, y, z)
    gl.glEnd()
    gl.glLineWidth(1.0)
    gl.glDisable(gl.GL_BLEND)


# ─────────────────────────────────────────────────────────────────────────────
# HUD (2-D overlay, no text — visual only)
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(width, height, drone_pos, roll, pitch, yaw,
             z_target, z_min, z_max):
    """
    Draw a minimal 2-D HUD:
      • Altitude bar  (right edge)  – yellow arrow = current, green tick = target
      • Attitude horizon strip      (bottom centre)
    """
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glPushMatrix()
    gl.glLoadIdentity()
    gl.glOrtho(0, width, 0, height, -1, 1)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glPushMatrix()
    gl.glLoadIdentity()
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    # ── Altitude bar ──────────────────────────────────────────────────────────
    bx = width - 50
    by = 60
    bw = 18
    bh = height - 120

    # Background
    gl.glColor4f(0, 0, 0, 0.45)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex2f(bx - 4, by - 4)
    gl.glVertex2f(bx + bw + 4, by - 4)
    gl.glVertex2f(bx + bw + 4, by + bh + 4)
    gl.glVertex2f(bx - 4, by + bh + 4)
    gl.glEnd()

    # Tick marks every 1 m
    gl.glColor4f(0.7, 0.7, 0.7, 0.5)
    gl.glLineWidth(1.0)
    for m in range(int(z_max) + 1):
        fy = by + (m / z_max) * bh
        gl.glBegin(gl.GL_LINES)
        gl.glVertex2f(bx, fy); gl.glVertex2f(bx + bw, fy)
        gl.glEnd()

    # Fill bar (current altitude)
    curr_frac   = float(np.clip((drone_pos[2] - z_min) / (z_max - z_min), 0, 1))
    target_frac = float(np.clip((z_target     - z_min) / (z_max - z_min), 0, 1))
    curr_y   = by + curr_frac   * bh
    target_y = by + target_frac * bh

    gl.glColor4f(0.3, 0.6, 1.0, 0.45)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex2f(bx, by); gl.glVertex2f(bx + bw, by)
    gl.glVertex2f(bx + bw, curr_y); gl.glVertex2f(bx, curr_y)
    gl.glEnd()

    # Target altitude – green horizontal line
    gl.glColor4f(0.1, 1.0, 0.1, 0.95)
    gl.glLineWidth(2.5)
    gl.glBegin(gl.GL_LINES)
    gl.glVertex2f(bx - 8, target_y)
    gl.glVertex2f(bx + bw + 8, target_y)
    gl.glEnd()

    # Current altitude – yellow arrow pointing right
    gl.glColor4f(1.0, 1.0, 0.1, 1.0)
    gl.glBegin(gl.GL_TRIANGLES)
    gl.glVertex2f(bx - 14, curr_y)
    gl.glVertex2f(bx - 3, curr_y + 7)
    gl.glVertex2f(bx - 3, curr_y - 7)
    gl.glEnd()

    # ── Artificial horizon strip (bottom centre) ──────────────────────────────
    cx = width // 2
    cy = 45
    hw = 160   # half-width
    hh = 28    # half-height

    # Panel background
    gl.glColor4f(0, 0, 0, 0.45)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex2f(cx - hw - 4, cy - hh - 4)
    gl.glVertex2f(cx + hw + 4, cy - hh - 4)
    gl.glVertex2f(cx + hw + 4, cy + hh + 4)
    gl.glVertex2f(cx - hw - 4, cy + hh + 4)
    gl.glEnd()

    # Sky (top half) / ground (bottom half) rotated by roll
    def rotated_rect(angle_rad, color_top, color_bot):
        s, c = math.sin(angle_rad), math.cos(angle_rad)
        # Horizon line endpoints
        lx1 = -hw * c - hh * s;  ly1 = -hw * s + hh * c
        lx2 =  hw * c - hh * s;  ly2 =  hw * s + hh * c
        lx3 =  hw * c + hh * s;  ly3 =  hw * s - hh * c
        lx4 = -hw * c + hh * s;  ly4 = -hw * s - hh * c
        # Sky quad
        gl.glColor3f(*color_top)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(cx + lx1, cy + ly1)
        gl.glVertex2f(cx + lx2, cy + ly2)
        gl.glVertex2f(cx + lx3, cy + ly3)
        gl.glVertex2f(cx + lx4, cy + ly4)
        gl.glEnd()

    # Draw ground half (brown) then sky half (blue) using stencil-like clip via
    # separate quads (approximate – no real clip needed for this size)
    # Simple approach: draw full rotated rect in sky colour, then overdraw lower half
    roll_disp = float(roll)

    s_r, c_r = math.sin(roll_disp), math.cos(roll_disp)

    def horizon_point(u):
        # u in [-hw, +hw] along horizon
        # pitch offsets horizon vertically
        pitch_pix = float(pitch) * 80.0   # pixels per radian
        return (cx + u * c_r + pitch_pix * s_r,
                cy + u * s_r - pitch_pix * c_r)

    # Sky
    gl.glColor3f(0.28, 0.52, 0.78)
    gl.glBegin(gl.GL_QUADS)
    hx1, hy1 = horizon_point(-hw); hx2, hy2 = horizon_point(hw)
    gl.glVertex2f(cx - hw, cy - hh)
    gl.glVertex2f(cx + hw, cy - hh)
    gl.glVertex2f(cx + hw, cy + hh)
    gl.glVertex2f(cx - hw, cy + hh)
    gl.glEnd()

    # Ground (below horizon)
    gl.glColor3f(0.40, 0.27, 0.13)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex2f(hx1, hy1)
    gl.glVertex2f(hx2, hy2)
    gl.glVertex2f(cx + hw, cy - hh)
    gl.glVertex2f(cx - hw, cy - hh)
    gl.glEnd()

    # Horizon line
    gl.glColor3f(1.0, 1.0, 1.0)
    gl.glLineWidth(1.8)
    gl.glBegin(gl.GL_LINES)
    gl.glVertex2f(hx1, hy1); gl.glVertex2f(hx2, hy2)
    gl.glEnd()

    # Centre aircraft symbol
    gl.glColor3f(1.0, 1.0, 0.0)
    gl.glLineWidth(2.0)
    gl.glBegin(gl.GL_LINES)
    gl.glVertex2f(cx - 30, cy); gl.glVertex2f(cx - 10, cy)
    gl.glVertex2f(cx + 10, cy); gl.glVertex2f(cx + 30, cy)
    gl.glVertex2f(cx, cy - 6); gl.glVertex2f(cx, cy + 6)
    gl.glEnd()

    # ── Controls legend strip (top-left) ─────────────────────────────────────
    leg_x, leg_y = 12, height - 12
    leg_w, leg_h = 215, 130
    gl.glColor4f(0, 0, 0, 0.45)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex2f(leg_x, leg_y - leg_h)
    gl.glVertex2f(leg_x + leg_w, leg_y - leg_h)
    gl.glVertex2f(leg_x + leg_w, leg_y)
    gl.glVertex2f(leg_x, leg_y)
    gl.glEnd()

    # Colour-coded key blocks (pure visuals – no font needed)
    def key_block(x, y, bw, bh, col):
        gl.glColor3f(*col)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(x, y); gl.glVertex2f(x + bw, y)
        gl.glVertex2f(x + bw, y + bh); gl.glVertex2f(x, y + bh)
        gl.glEnd()

    ks = 14  # key size
    km = 5   # margin
    row_h = ks + km
    bx0 = leg_x + 10
    by0 = leg_y - 20

    labels = [
        # col,         label_col
        ((0.9, 0.6, 0.1), "W/S – Pitch"),
        ((0.2, 0.7, 0.9), "A/D – Roll"),
        ((0.8, 0.3, 0.8), "Q/E – Yaw"),
        ((0.2, 1.0, 0.2), "↑↓  – Alt target"),
        ((0.9, 0.9, 0.2), "←→  – Cam orbit"),
    ]

    for k, (col, _) in enumerate(labels):
        key_block(bx0, by0 - k * row_h - ks, ks, ks, col)

    # ── Status bar: roll / pitch / yaw / alt compact bar ─────────────────────
    sb_x, sb_y = width // 2 - 130, height - 14
    sb_w, sb_h = 260, 22

    gl.glColor4f(0, 0, 0, 0.45)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex2f(sb_x, sb_y - sb_h)
    gl.glVertex2f(sb_x + sb_w, sb_y - sb_h)
    gl.glVertex2f(sb_x + sb_w, sb_y)
    gl.glVertex2f(sb_x, sb_y)
    gl.glEnd()

    # Attitude needles – small coloured bars that fill proportionally
    def attitude_gauge(x, y, w, h, value, v_max, col):
        frac = float(np.clip(value / v_max, -1, 1))
        cx2 = x + w // 2
        gl.glColor4f(0.3, 0.3, 0.3, 0.6)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(x, y); gl.glVertex2f(x+w, y)
        gl.glVertex2f(x+w, y+h); gl.glVertex2f(x, y+h)
        gl.glEnd()
        gl.glColor3f(*col)
        blen = (w // 2) * abs(frac)
        if frac >= 0:
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex2f(cx2, y+2); gl.glVertex2f(cx2+blen, y+2)
            gl.glVertex2f(cx2+blen, y+h-2); gl.glVertex2f(cx2, y+h-2)
            gl.glEnd()
        else:
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex2f(cx2-blen, y+2); gl.glVertex2f(cx2, y+2)
            gl.glVertex2f(cx2, y+h-2); gl.glVertex2f(cx2-blen, y+h-2)
            gl.glEnd()
        # Centre tick
        gl.glColor3f(1, 1, 1)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex2f(cx2, y); gl.glVertex2f(cx2, y+h)
        gl.glEnd()

    gauge_w = 72
    gap = 8
    gx = sb_x + 10
    gy = sb_y - sb_h + 3
    gh = sb_h - 6

    attitude_gauge(gx,                      gy, gauge_w, gh, float(roll),  math.pi/4, (0.9, 0.4, 0.2))
    attitude_gauge(gx + gauge_w + gap,       gy, gauge_w, gh, float(pitch), math.pi/4, (0.2, 0.7, 0.9))
    attitude_gauge(gx + 2*(gauge_w + gap),   gy, gauge_w, gh, float(yaw),   math.pi,   (0.8, 0.3, 0.8))

    gl.glDisable(gl.GL_BLEND)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glLineWidth(1.0)
    gl.glPopMatrix()
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glPopMatrix()
    gl.glMatrixMode(gl.GL_MODELVIEW)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not glfw.init():
        return

    window = glfw.create_window(WIDTH, HEIGHT, "6-DOF INDI Drone Simulator", None, None)
    glfw.make_context_current(window)

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_LINE_SMOOTH)
    gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gluPerspective(52, WIDTH / HEIGHT, 0.05, 300)
    gl.glMatrixMode(gl.GL_MODELVIEW)

    # ── Simulation objects ────────────────────────────────────────────────────
    drone      = DroneDynamics()
    controller = INDIController(drone)

    drone.position = np.array([0.0, 0.0, 0.5])
    drone.velocity = np.zeros(3)

    # Initial motor spin-up so INDI has a non-zero tau_prev baseline
    omega_hover = np.sqrt(controller.hover_T / drone.kT)
    drone.Omega     = np.ones(4) * omega_hover
    drone.Omega_cmd = np.ones(4) * omega_hover

    dt = 0.005

    # ── Flight limits ─────────────────────────────────────────────────────────
    MAX_ROLL_CMD  = 0.30   # rad  (~17°)
    MAX_PITCH_CMD = 0.30
    MAX_YAW_RATE  = 2.00   # rad/s

    # ── Altitude target ───────────────────────────────────────────────────────
    Z_TARGET_INIT = 2.0
    Z_MIN, Z_MAX  = 0.3, 8.0
    Z_STEP        = 0.5
    z_target      = Z_TARGET_INIT

    # Edge-detection state for altitude keys
    key_up_prev   = False
    key_down_prev = False

    # ── Camera state ──────────────────────────────────────────────────────────
    cam_azimuth   = 45.0    # degrees, orbiting around drone
    cam_elevation = 22.0    # degrees above horizontal
    cam_distance  = 5.0

    # Smoothed camera target (lag-filtered drone position)
    cam_target = drone.position.copy()

    # ── Simulation clock ─────────────────────────────────────────────────────
    sim_time = 0.0
    Omega_cmd = np.ones(4) * omega_hover
    frame = 0

    # ── Altitude PD gains ────────────────────────────────────────────────────
    Kp_z = 12.0
    Kd_z = 4.0

    print("Controls:  W/S=pitch  A/D=roll  Q/E=yaw  ↑↓=alt target  ←→=cam orbit")
    print(f"Initial altitude target: {z_target:.1f} m")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while not glfw.window_should_close(window):

        # ── Background (sky gradient approximated by clear colour) ────────────
        gl.glClearColor(0.42, 0.63, 0.82, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # ── Camera ────────────────────────────────────────────────────────────
        # Smoothly track drone
        alpha_cam = 0.04
        cam_target += alpha_cam * (drone.position - cam_target)

        az  = math.radians(cam_azimuth)
        el  = math.radians(cam_elevation)
        eye = cam_target + cam_distance * np.array([
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ])

        gl.glLoadIdentity()
        gluLookAt(*eye, *cam_target, 0, 0, 1)

        # ── Keyboard input ────────────────────────────────────────────────────
        def key(k):
            return glfw.get_key(window, k) == glfw.PRESS

        roll_cmd     = 0.0
        pitch_cmd    = 0.0
        yaw_rate_cmd = 0.0

        if key(glfw.KEY_D): roll_cmd     = -MAX_ROLL_CMD
        if key(glfw.KEY_A): roll_cmd     =  MAX_ROLL_CMD
        if key(glfw.KEY_W): pitch_cmd    = -MAX_PITCH_CMD
        if key(glfw.KEY_S): pitch_cmd    =  MAX_PITCH_CMD
        if key(glfw.KEY_Q): yaw_rate_cmd =  MAX_YAW_RATE
        if key(glfw.KEY_E): yaw_rate_cmd = -MAX_YAW_RATE

        # Altitude target – step on leading edge only (no auto-repeat)
        up_now   = key(glfw.KEY_UP)
        down_now = key(glfw.KEY_DOWN)

        if up_now   and not key_up_prev:
            z_target = min(z_target + Z_STEP, Z_MAX)
            print(f"Alt target → {z_target:.1f} m")
        if down_now and not key_down_prev:
            z_target = max(z_target - Z_STEP, Z_MIN)
            print(f"Alt target → {z_target:.1f} m")

        key_up_prev   = up_now
        key_down_prev = down_now

        # Camera orbit
        if key(glfw.KEY_LEFT):  cam_azimuth -= 0.8
        if key(glfw.KEY_RIGHT): cam_azimuth += 0.8

        # ── Altitude PD ───────────────────────────────────────────────────────
        z_err      = z_target - drone.position[2]
        vz_err     = -drone.velocity[2]
        thrust_cmd = drone.m * drone.g + Kp_z * z_err + Kd_z * vz_err
        thrust_cmd = float(np.clip(
            thrust_cmd,
            0.25 * drone.m * drone.g,
            1.85 * drone.m * drone.g
        ))

        # ── Attitude + rate INDI controller ───────────────────────────────────
        Omega_cmd = controller.compute_rpm(
            roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd
        )
        drone.Omega_cmd = Omega_cmd
        drone.step(dt)

        sim_time += dt
        frame    += 1

        roll, pitch, yaw = controller.quat_to_euler(drone.q)

        if frame % 40 == 0:
            print(
                f"z={drone.position[2]:.2f}/{z_target:.1f}m  "
                f"roll={math.degrees(roll):+6.1f}°  "
                f"pitch={math.degrees(pitch):+6.1f}°  "
                f"yaw={math.degrees(yaw):+6.1f}°  "
                f"Ω̄={np.mean(Omega_cmd):.0f} rad/s"
            )

        # ── Scene rendering ───────────────────────────────────────────────────

        # Ground grid
        draw_grid(size=20, step=1)

        # World axes at origin
        draw_world_axes()

        # Drone shadow
        draw_shadow(drone.position)

        # Vertical reference line under drone
        draw_drone_altitude_line(drone.position)

        # Target altitude indicator (follows drone XY)
        pulse = math.sin(sim_time * 3.0)
        draw_altitude_target(z_target, drone.position[0], drone.position[1], pulse)

        # Drone model (in body frame)
        gl.glPushMatrix()
        gl.glTranslatef(*drone.position)
        R = quat_to_matrix(drone.q)
        M = np.eye(4)
        M[:3, :3] = R
        gl.glMultMatrixf(M.T.flatten())
        draw_drone_model(Omega_cmd, sim_time)
        gl.glPopMatrix()

        # 2-D HUD overlay
        draw_hud(WIDTH, HEIGHT, drone.position,
                 roll, pitch, yaw,
                 z_target, Z_MIN, Z_MAX)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()