import pygame
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import resample

# =============================================
# PARÁMETROS FÍSICOS
# =============================================
v_sound = 343.0      # m/s
f_emit = 440.0       # Hz de referencia

# =============================================
# CONFIGURACIÓN PANTALLA
# =============================================
WIDTH, HEIGHT = 900, 700
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

# Cargar archivo de sirena
sirena = pygame.mixer.Sound("sirena.wav")

# Crear canal de reproducción
channel = pygame.mixer.Channel(0)
channel.play(sirena, loops=-1)  # loop infinito

# =============================================
# CARGAR IMÁGENES
# =============================================
ambulance_img = pygame.image.load("ambulancia.png")
ambulance_img = pygame.transform.scale(ambulance_img, (100, 50))

observer_img = pygame.image.load("persona.png")
observer_img = pygame.transform.scale(observer_img, (60, 60))

# =============================================
# OBJETOS
# =============================================
source_pos = np.array([200.0, 350.0])
observer_pos = np.array([700.0, 350.0])

dragging_source = False
dragging_observer = False

vel_s = np.array([0.0, 0.0])

# =============================================
# HISTORIA PARA GRÁFICAS
# =============================================
history_t = deque(maxlen=600)
history_d = deque(maxlen=600)
history_f = deque(maxlen=600)

time_elapsed = 0.0
last_pitch_update = 0.0

# =============================================
# ONDAS
# =============================================
rings = []
ring_interval = 0.25
last_ring_time = 0.0

# =============================================
# FUNCIONES
# =============================================
def compute_distance(p1, p2):
    r = p1 - p2
    return np.linalg.norm(r), r

def compute_radial_velocity(r_vec):
    d = np.linalg.norm(r_vec)
    if d == 0:
        return 0.0
    r_hat = r_vec / d
    return np.dot(vel_s, r_hat)

def doppler_freq(f_emit, v_sound, v_r):
    denom = v_sound - v_r
    if np.isclose(denom, 0):
        return f_emit
    return f_emit * (v_sound / denom)

# =============================================
# CONFIG GRÁFICAS
# =============================================
plt.ion()
fig, (ax_d, ax_f) = plt.subplots(2,1, figsize=(7,8))
fig.tight_layout(pad=4)

line_d, = ax_d.plot([], [], 'b-')
line_f, = ax_f.plot([], [], 'r-')

ax_d.set_ylabel("Distancia d(t)")
ax_f.set_ylabel("Frecuencia f'(t)")
ax_f.set_xlabel("Tiempo (s)")

# =============================================
# LOOP PRINCIPAL
# =============================================
running = True

while running:
    dt = clock.tick(60) / 1000.0
    time_elapsed += dt

    # --------------------------------------
    # EVENTOS
    # --------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if np.linalg.norm(source_pos - np.array([mx, my])) < 50:
                dragging_source = True
            if np.linalg.norm(observer_pos - np.array([mx, my])) < 40:
                dragging_observer = True

        if event.type == pygame.MOUSEBUTTONUP:
            dragging_source = False
            dragging_observer = False

    # --------------------------------------
    # MOVER OBJETOS
    # --------------------------------------
    prev_source = source_pos.copy()

    if dragging_source:
        source_pos = np.array(pygame.mouse.get_pos(), float)
    if dragging_observer:
        observer_pos = np.array(pygame.mouse.get_pos(), float)

    vel_s = (source_pos - prev_source) / max(dt, 1e-6)

    # --------------------------------------
    # CÁLCULO DOPPLER
    # --------------------------------------
    d, r_vec = compute_distance(source_pos, observer_pos)
    v_r = compute_radial_velocity(r_vec)
    f_obs = doppler_freq(f_emit, v_sound, v_r)

    # VOLUMEN ~ 1/d²
    volume = min(1.0, 3000.0/(d**2 + 1))
    channel.set_volume(volume)

    # PITCH DINÁMICO (cada 0.15s)
    if time_elapsed - last_pitch_update > 0.15:
        ratio = f_obs / f_emit
        ratio = max(0.5, min(ratio, 2.0))

        arr = pygame.sndarray.array(sirena)
        small = arr[:4000]
        new_n = int(len(small)/ratio)

        # remuestrear
        new_small = np.ascontiguousarray(
            resample(small, new_n).astype(np.int16)
        )

        snd = pygame.sndarray.make_sound(new_small)
        channel.queue(snd)

        last_pitch_update = time_elapsed

    # --------------------------------------
    # ONDAS
    # --------------------------------------
    if time_elapsed - last_ring_time > ring_interval:
        rings.append((source_pos.copy(), time_elapsed))
        last_ring_time = time_elapsed

    # --------------------------------------
    # DIBUJO
    # --------------------------------------
    window.fill((30,30,30))

    for pos0, t0 in rings:
        age = time_elapsed - t0
        center = pos0 - vel_s * age
        radius = v_sound * age
        if radius < max(WIDTH, HEIGHT):
            pygame.draw.circle(window, (120,120,255),
                               center.astype(int), int(radius), 2)

    window.blit(ambulance_img, (source_pos[0]-50, source_pos[1]-25))
    window.blit(observer_img, (observer_pos[0]-30, observer_pos[1]-30))

    pygame.draw.line(window, (255,255,255),
                     source_pos.astype(int),
                     observer_pos.astype(int), 3)

    # ==================================================
    # SECCIÓN 2: VARIABLES FÍSICAS
    # ==================================================
    vars_rect = pygame.Rect(0, HEIGHT-200, WIDTH//2, 200)
    pygame.draw.rect(window, (10,10,10), vars_rect)

    def show_var(label, val, ypos):
        txt = font.render(f"{label}: {val}", True, (200,200,200))
        window.blit(txt, (10, ypos))

    show_var("f",        f"{f_emit:.2f} Hz",          HEIGHT-190)
    show_var("f'(t)",    f"{f_obs:.2f} Hz",           HEIGHT-165)
    show_var("v",        f"{v_sound:.1f} m/s",        HEIGHT-140)
    show_var("ps(t)",    f"{np.round(source_pos,1)}", HEIGHT-115)
    show_var("po(t)",    f"{np.round(observer_pos,1)}", HEIGHT-90)
    show_var("vs",       f"{np.round(vel_s,1)} m/s",  HEIGHT-65)
    show_var("d(t)",     f"{d:.1f} m",                HEIGHT-40)
    show_var("vr(t)",    f"{v_r:.1f} m/s",            HEIGHT-15)

    # ==================================================
    # SECCIÓN 3: FÓRMULA DOPPLER CON VALORES
    # ==================================================
    form_rect = pygame.Rect(WIDTH//2, HEIGHT-200, WIDTH//2, 200)
    pygame.draw.rect(window, (20,20,20), form_rect)

    # Línea con fórmula
    num1 = f"{f_emit:.2f} · {v_sound:.1f}"
    num2 = f"{v_sound:.1f} - {v_r:.2f}"

    window.blit(font.render("f'(t) =", True, (255,255,0)),
                (WIDTH//2+10, HEIGHT-190))

    window.blit(font.render(f"({num1}) / ({num2})", True, (255,255,0)),
                (WIDTH//2+80, HEIGHT-190))

    # Resultado
    window.blit(font.render(f"= {f_obs:.2f} Hz", True, (0,255,0)),
                (WIDTH//2+10, HEIGHT-160))

    pygame.display.flip()

    # --------------------------------------
    # GRÁFICAS
    # --------------------------------------
    history_t.append(time_elapsed)
    history_d.append(d)
    history_f.append(f_obs)

    line_d.set_xdata(history_t)
    line_d.set_ydata(history_d)

    line_f.set_xdata(history_t)
    line_f.set_ydata(history_f)

    ax_d.set_xlim(max(0,time_elapsed-6), time_elapsed)
    ax_f.set_xlim(max(0,time_elapsed-6), time_elapsed)

    ax_d.set_ylim(0, max(history_d)+20)
    ax_f.set_ylim(200, 1000)

    plt.pause(0.001)

# =============================================
# SALIDA SEGURA (FUERA DEL WHILE)
# =============================================
try:
    channel.stop()
    pygame.mixer.stop()
    pygame.mixer.quit()
except:
    pass

try:
    pygame.quit()
except:
    pass
