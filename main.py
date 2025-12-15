import pygame
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def draw_button(surface, rect, text, hover=False, active=False):
    base = (50, 200, 50)
    hovered = (70, 230, 70)
    active_c = (200, 50, 50)

    color = active_c if active else (hovered if hover else base)

    pygame.draw.rect(surface, color, rect, border_radius=10)
    pygame.draw.rect(surface, (0, 0, 0), rect, width=2, border_radius=10)

    txt = font.render(text, True, (0, 0, 0))
    t_rect = txt.get_rect(center=rect.center)
    surface.blit(txt, t_rect)


# =============================================
# PARÁMETROS FÍSICOS
# =============================================
v_sound = 343.0      # m/s
f_emit = 440.0       # Hz de referencia

# =============================================
# CONFIGURACIÓN PANTALLA
# =============================================
WIDTH, HEIGHT = 1100, 850
pygame.init()
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulador Doppler (sirena.wav)")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# =============================================
# AUDIO
# =============================================
pygame.mixer.quit()
pygame.mixer.init(frequency=44100, size=-16, channels=2)

sirena = pygame.mixer.Sound("Utilities/sirena.wav")
channel = pygame.mixer.Channel(0)
channel.play(sirena, loops=-1)  # loop infinito

# =============================================
# CARGAR IMÁGENES
# =============================================
ambulance_img = pygame.image.load("Utilities/ambulancia.png")
ambulance_img = pygame.transform.scale(ambulance_img, (100, 50))

observer_img = pygame.image.load("Utilities/persona.png")
observer_img = pygame.transform.scale(observer_img, (60, 60))

background = pygame.image.load("Utilities/fondo.jpg")
background = pygame.transform.scale(background, (WIDTH, HEIGHT))

# =============================================
# OBJETOS
# =============================================
source_pos = np.array([200.0, 350.0])
observer_pos = np.array([700.0, 350.0])

dragging_source = False
dragging_observer = False

vel_s = np.array([0.0, 0.0])
vel_o = np.array([0.0, 0.0])

# =============================================
# HISTORIA PARA GRÁFICAS
# =============================================
history_t = deque(maxlen=600)
history_d = deque(maxlen=600)
history_f = deque(maxlen=600)

time_elapsed = 0.0

# =============================================
# ONDAS
# =============================================
rings = []
ring_interval = 0.25
last_ring_time = 0.0

flash_time = 0.005
flash_duration = 0.05

auto_source = False
auto_observer = False
paused = False

# =============================================
# FUNCIONES
# =============================================
def compute_distance(p1, p2):
    r = p1 - p2
    return np.linalg.norm(r), r

def doppler_freq_general(f_emit, v_sound, vs_rad, vo_rad):
    denom = v_sound - vs_rad
    if np.isclose(denom, 0):
        return f_emit
    return (v_sound + vo_rad) / denom * f_emit

# =============================================
# CONFIG GRÁFICAS
# =============================================
plt.ion()
fig, (ax_d, ax_f) = plt.subplots(2, 1, figsize=(7, 8))
fig.tight_layout(pad=4)

line_d, = ax_d.plot([], [], 'b-')
line_f, = ax_f.plot([], [], 'r-')

ax_d.set_ylabel("Distancia d(t)")
ax_f.set_ylabel("Frecuencia f'(t)")
ax_f.set_xlabel("Tiempo (s)")

ax_d.set_title("Evolución de la distancia fuente-observador")
ax_f.set_title("Variación de la frecuencia observada (Efecto Doppler)")

# =============================================
# LOOP PRINCIPAL
# =============================================
pause_button = pygame.Rect(10, 10, 140, 40)
running = True

while running:
    dt = clock.tick(60) / 1000.0

    # --------------------------------------
    # EVENTOS
    # --------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                paused = not paused
            if event.key == pygame.K_a:
                auto_source = not auto_source
            if event.key == pygame.K_o:
                auto_observer = not auto_observer

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if pause_button.collidepoint(mx, my):
                paused = not paused
            if np.linalg.norm(source_pos - np.array([mx, my])) < 50:
                dragging_source = True
            if np.linalg.norm(observer_pos - np.array([mx, my])) < 40:
                dragging_observer = True

        if event.type == pygame.MOUSEBUTTONUP:
            dragging_source = False
            dragging_observer = False

    # --------------------------------------
    # PAUSA
    # --------------------------------------
    if paused:
        if channel.get_busy():
            channel.pause()

        window.blit(background, (0, 0))

        for pos0, t0 in rings:
            age = time_elapsed - t0
            center = pos0 - vel_s * age
            radius = v_sound * age
            if radius < max(WIDTH, HEIGHT):
                pygame.draw.circle(window, (120, 120, 255),
                                   center.astype(int), int(radius), 2)

        window.blit(ambulance_img, (source_pos[0] - 50, source_pos[1] - 25))
        window.blit(observer_img, (observer_pos[0] - 30, observer_pos[1] - 30))

        pygame.draw.line(window, (255, 255, 255),
                         source_pos.astype(int),
                         observer_pos.astype(int), 3)

        mx, my = pygame.mouse.get_pos()
        hover = pause_button.collidepoint(mx, my)
        draw_button(window, pause_button, "REANUDAR", hover, active=True)

        pygame.display.flip()
        continue

    if not channel.get_busy():
        channel.play(sirena, loops=-1)
    else:
        try:
            channel.unpause()
        except:
            pass

    # --------------------------------------
    # AVANZAR TIEMPO
    # --------------------------------------
    time_elapsed += dt

    # --------------------------------------
    # MOVER OBJETOS
    # --------------------------------------
    prev_source = source_pos.copy()
    prev_observer = observer_pos.copy()

    if dragging_source:
        source_pos = np.array(pygame.mouse.get_pos(), float)
        # suavizar arrastre
        source_pos = prev_source + (source_pos - prev_source) * 0.2

    if dragging_observer:
        observer_pos = np.array(pygame.mouse.get_pos(), float)
        observer_pos = prev_observer + (observer_pos - prev_observer) * 0.2

    if auto_source:
        source_pos += np.array([100 * dt, 0])

    if auto_observer:
        observer_pos += np.array([-80 * dt, 0])

    vel_s = (source_pos - prev_source) / max(dt, 1e-6)
    vel_o = (observer_pos - prev_observer) / max(dt, 1e-6)

    # ===============================
    # LIMITAR VELOCIDADES A SUBSÓNICAS
    # ===============================
    max_speed = 50.0  # m/s (~180 km/h)

    speed_s = np.linalg.norm(vel_s)
    speed_o = np.linalg.norm(vel_o)

    if speed_s > max_speed:
        vel_s = vel_s * (max_speed / speed_s)

    if speed_o > max_speed:
        vel_o = vel_o * (max_speed / speed_o)

    # --------------------------------------
    # CÁLCULO DOPPLER
    # --------------------------------------
    d, r_vec = compute_distance(source_pos, observer_pos)

    if d == 0:
        r_hat = np.array([0.0, 0.0])
    else:
        r_hat = r_vec / d

    vs_proj = np.dot(vel_s, r_hat)
    vo_proj = np.dot(vel_o, r_hat)

    vs_rad = -vs_proj
    vo_rad = vo_proj

    f_obs = doppler_freq_general(f_emit, v_sound, vs_rad, vo_rad)

    # COLOR SEGÚN DOPPLER
    if abs(f_obs - f_emit) < 1.0:
        wave_color = (120,120,255)
    elif f_obs > f_emit:
        wave_color = (0,255,0)
    else:
        wave_color = (255,0,0)ñ

    # --------------------------------------
    # VOLUMEN
    # --------------------------------------
    base_vol = 3000.0 / (d**2 + 1)
    volume = max(0.05, min(1.0, base_vol))
    channel.set_volume(volume)

    # --------------------------------------
    # ONDAS
    # --------------------------------------
    if time_elapsed - last_ring_time > ring_interval:
        rings.append((source_pos.copy(), time_elapsed))
        last_ring_time = time_elapsed

    # --------------------------------------
    # DIBUJO
    # --------------------------------------
    window.blit(background, (0, 0))

    for pos0, t0 in rings:
        age = time_elapsed - t0
        center = pos0 - vel_s * age
        radius = v_sound * age
        if radius < max(WIDTH, HEIGHT):
            pygame.draw.circle(window, wave_color,
                               center.astype(int), int(radius), 2)
            if np.linalg.norm(center - observer_pos) < radius + 3:
                flash_time = time_elapsed

    window.blit(ambulance_img, (source_pos[0]-50, source_pos[1]-25))

    if (time_elapsed - flash_time) < flash_duration:
        flash_img = observer_img.copy()
        flash_img.fill((255,255,255), None, pygame.BLEND_ADD)
        window.blit(flash_img, (observer_pos[0]-30, observer_pos[1]-30))
    else:
        window.blit(observer_img, (observer_pos[0]-30, observer_pos[1]-30))

    pygame.draw.line(window, (255,255,255),
                     source_pos.astype(int),
                     observer_pos.astype(int), 3)

    mx, my = pygame.mouse.get_pos()
    hover = pause_button.collidepoint(mx, my)
    draw_button(window, pause_button, "PAUSAR", hover, paused)

    # =============================================
    # SECCIÓN VARIABLES FÍSICAS
    # =============================================
    vars_rect = pygame.Rect(0, HEIGHT - 260, WIDTH // 2, 260)
    pygame.draw.rect(window, (15,15,15), vars_rect)
    pygame.draw.rect(window, (250,250,250), vars_rect, 2)

    window.blit(font.render("VARIABLES FÍSICAS", True, (255,255,0)),
                (10, HEIGHT - 245))

    line_y = HEIGHT - 220

    def draw_line(label, val):
        txt = font.render(f"{label}: {val}", True, (230,230,230))
        window.blit(txt, (20, draw_line.y))
        draw_line.y += 22

    draw_line.y = line_y

    draw_line("Pos fuente ps(t)", f"{np.round(source_pos, 1)}")
    draw_line("Pos observador po(t)", f"{np.round(observer_pos, 1)}")
    draw_line("Frecuencia emitida f", f"{f_emit:.2f} Hz")
    draw_line("Frecuencia observada f'(t)", f"{f_obs:.2f} Hz")
    draw_line("Velocidad sonido v", f"{v_sound:.1f} m/s")
    draw_line("||vs||", f"{np.linalg.norm(vel_s):.2f} m/s")
    draw_line("||vo||", f"{np.linalg.norm(vel_o):.2f} m/s")
    draw_line("Distancia d(t)", f"{d:.1f} m")
    draw_line("vs_rad", f"{vs_rad:.2f} m/s")
    draw_line("vo_rad", f"{vo_rad:.2f} m/s")

    # =============================================
    # FORMULA DOPPLER
    # =============================================
    form_rect = pygame.Rect(WIDTH // 2, HEIGHT - 260, WIDTH // 2, 260)
    pygame.draw.rect(window, (15,15,15), form_rect)
    pygame.draw.rect(window, (250,250,250), form_rect, 2)

    title = font.render("FÓRMULA DOPPLER GENERAL", True, (255,255,0))
    window.blit(title, (WIDTH // 2 + 10, HEIGHT - 235))

    num = f"{v_sound:.1f} + ({vo_rad:.2f})"
    den = f"{v_sound:.1f} - ({vs_rad:.2f})"

    num_txt = font.render(num, True, (255,255,255))
    den_txt = font.render(den, True, (255,255,255))
    emit_txt = font.render(f"{f_emit:.2f}", True, (255,255,255))

    cx = WIDTH // 2 + 150
    cy = HEIGHT - 170

    window.blit(num_txt, (cx - num_txt.get_width()//2, cy - 20))
    pygame.draw.line(window, (255,255,255), (cx - 80, cy), (cx + 80, cy), 2)
    window.blit(den_txt, (cx - den_txt.get_width()//2, cy + 5))

    times_txt = font.render("×", True, (255,255,255))
    window.blit(times_txt, (cx + 100, cy - 5))
    window.blit(emit_txt, (cx + 130, cy - 5))

    res_txt = font.render(f"f'(t) = {f_obs:.2f} Hz", True, (0,255,0))
    window.blit(res_txt, (WIDTH // 2 + 10, HEIGHT - 85))

    theta = np.degrees(np.arctan2(r_hat[1], r_hat[0])) if d > 0 else 0
    window.blit(font.render(f"θ(t) = {theta:.1f}°", True, (255,255,0)),
                (WIDTH // 2 + 10, HEIGHT - 60))

    pygame.display.flip()

    # =============================================
    # GRÁFICAS
    # =============================================
    history_t.append(time_elapsed)
    history_d.append(d)
    history_f.append(f_obs)

    line_d.set_xdata(history_t)
    line_d.set_ydata(history_d)

    line_f.set_xdata(history_t)
    line_f.set_ydata(history_f)

    ax_d.set_xlim(max(0, time_elapsed - 6), time_elapsed)
    ax_f.set_xlim(max(0, time_elapsed - 6), time_elapsed)

    ax_d.set_ylim(0, max(history_d) + 20)
    ax_f.set_ylim(200, 1000)

    plt.pause(0.001)

try:
    channel.stop()
except:
    pass

try:
    pygame.mixer.quit()
except:
    pass

try:
    pygame.quit()
except:
    pass
