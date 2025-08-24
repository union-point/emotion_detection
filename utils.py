import torch


def noise_tensor(iteration, noise_scale=0.1, base_seed=42):
    tensor = torch.tensor(
        [
            [-0.1187, 0.1340, -0.0531, -0.1008, 0.1027, 0.0046, -0.0252],
            [-0.0961, -0.0169, -0.2607, -0.1972, 0.2431, 0.0258, -0.1594],
        ]
    )

    original_state = torch.random.get_rng_state()

    try:
        # Устанавливаем seed на основе итерации
        torch.manual_seed(base_seed + iteration)

        # Генерируем шум
        noise = torch.randn_like(tensor) * noise_scale

        # Возвращаем тензор с шумом
        return tensor + noise

    finally:
        # Восстанавливаем оригинальное состояние генератора
        torch.random.set_rng_state(original_state)
