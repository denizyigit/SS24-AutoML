import neps

pipeline_space = {
    "batch_size": neps.Categorical(choices=[32.0, 64.0, 128.0, 256.0], default=64.0, default_confidence="low"),
    "learning_rate": neps.Float(lower=1e-6, upper=1e-1, log=True, default=1e-3, default_confidence="medium"),
    "epochs": neps.Integer(lower=1, upper=2, is_fidelity=True),
    "optimizer": neps.Categorical(choices=["adam", "sgd"], default="adam", default_confidence="low"),
    "momentum": neps.Float(lower=0.1, upper=0.999, default=0.4, default_confidence="low"),
    "weight_dec_active": neps.Categorical(choices=["True", "False"], default="False", default_confidence="low"),
    "weight_dec_adam": neps.Float(lower=0.00001, upper=0.1, default=1e-4, default_confidence="low"),
    "weight_dec_sgd": neps.Float(lower=0.00001, upper=0.1, default=1e-4, default_confidence="low"),

    # Data Augmentation
    "random_horizontal_flip_prob": neps.Float(lower=0.0, upper=1.0, default=0.5, default_confidence="low"),
    "random_vertical_flip_prob": neps.Float(lower=0.0, upper=1.0, default=0.5, default_confidence="low"),
    "random_rotation_deg": neps.Integer(lower=0, upper=180, default=90, default_confidence="low"),
    "random_rotation_prob": neps.Float(lower=0.0, upper=1.0, default=0.5, default_confidence="low"),
    "random_gaussian_noise_mean": neps.Float(lower=0.0, upper=1.0, default=0.0, default_confidence="low"),
    "random_gaussian_noise_std": neps.Float(lower=0.0, upper=1.0, default=0.1, default_confidence="low"),
    "random_gaussian_noise_prob": neps.Float(lower=0.0, upper=1.0, default=0.5, default_confidence="low"),
    "brightness": neps.Float(lower=0.0, upper=1.0, default=0.5, default_confidence="low"),
    "contrast": neps.Float(lower=0.0, upper=1.0, default=0.5, default_confidence="low"),
    "saturation": neps.Float(lower=0.0, upper=1.0, default=0.5, default_confidence="low"),
}
