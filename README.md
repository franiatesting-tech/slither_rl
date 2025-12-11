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
