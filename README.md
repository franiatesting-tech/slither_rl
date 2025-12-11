# slither_rl

Proyecto de investigación y desarrollo de agentes de Deep Reinforcement Learning para un entorno tipo Slither.io.

Hito 1:
- Entorno simulado `SlitherSimEnv` basado en Gymnasium.
- Observaciones iniciales como *features* compactos.
- Agente baseline PPO usando Stable-Baselines3.

## Hito 2: Entorno de *features* y baselines

### Entorno `SlitherSim-v0`

- **Observación**: espacio `Box(shape=(obs_dim,), dtype=float32)` con las *features*:
  - `pos_x`, `pos_y` normalizados en `[-1, 1]`.
  - `cos(theta)`, `sin(theta)`.
  - `length_norm`.
  - `dist_food_norm`, `angle_food_norm` (comida más cercana).
  - `dist_border_norm`.
- **Acciones**: `Discrete(4)`
  - `0`: girar izquierda.
  - `1`: recto.
  - `2`: girar derecha.
  - `3`: *boost*.
- **Reward config B** (ajustada contra baseline random):
  - `step_penalty = 0.02` (coste pequeño por vivir).
  - `food_reward = 2.0` (recompensa moderada por comida).
  - `death_penalty = 30.0` (castigo grande al morir).

Entrenamiento PPO con Stable-Baselines3 (8 entornos en paralelo, `PPO("MlpPolicy")`, ~2e5 *timesteps*).

### Resultados actuales (`SlitherSim-v0`, `obs_type="features"`)

- **Random policy**: `mean_reward = -29.9 ± 1.1`.
- **Greedy food baseline** (giro miope hacia la comida): `mean_reward = -11.6 ± 12.6`.
- **PPO (SB3)**: `mean_reward = -3.3 ± 1.1`.

PPO supera claramente a las políticas random y heurística *greedy*, y las curvas de TensorBoard muestran un incremento de la longitud media del episodio (`ep_len_mean`) de ~20 a ~60–70 pasos.

## Hito 3 – Comparativa features vs radar

**Configuración:**
- Entorno `SlitherSim-v0` (`obs_type="features"`, `max_steps=200`).
- Entorno `SlitherSimRadar-v0` (`obs_type="radar"`, `N_SECTORS=8`, `obs_dim=19`).
- Algoritmo: PPO (Stable-Baselines3), 8 entornos en paralelo, 300k *timesteps* por *seed*.
- 3 *seeds* por configuración (features / radar).

**Baselines (50 episodios):**
- Random: `−30.14 ± 1.05`.
- Greedy hacia comida: `−11.96 ± 13.92`.

**Resultados PPO (multi-seed, 50 episodios por evaluación):**

| Policy       | Env                | Obs      | Cross-seed mean ± std |
|--------------|--------------------|----------|------------------------|
| PPO_features | SlitherSim-v0      | features | −2.59 ± 0.36           |
| PPO_radar    | SlitherSimRadar-v0 | radar    | +9.71 ± 1.24           |

**Interpretación:**
- PPO_features aprende a sobrevivir casi todo el horizonte (`ep_len_mean ≈ 200` por el tope de `max_steps`), pero solo consigue una recompensa ligeramente negativa.
- PPO_radar no solo alcanza el máximo de pasos, sino que obtiene recompensa claramente positiva (+9.7), superando ampliamente a las baselines (random/greedy) y a features.
- Las curvas de TensorBoard muestran saturación de `rollout/ep_len_mean` alrededor de 200 por el límite de `max_steps`, mientras que `rollout/ep_rew_mean` sigue mejorando en el caso de radar.
