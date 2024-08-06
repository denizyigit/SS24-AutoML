import neps


class PipelineSpace:
    def __init__(self):
        self.batch_size = neps.Categorical(
            choices=[32.0, 64.0, 128.0, 256.0], default=64.0, default_confidence="low")

        self.learning_rate = neps.Float(
            lower=1e-6, upper=1e-1, log=True, default=1e-3, default_confidence="medium")

        self.epochs = neps.Integer(lower=1, upper=15, is_fidelity=True)

        # Scheduler
        self.scheduler = neps.Categorical(
            choices=["oneCycleLR", "reduceLROnPlateau"], default="oneCycleLR", default_confidence="low")

        #  Optimizer
        self.optimizer = neps.Categorical(
            choices=["adam", "sgd"], default="adam", default_confidence="medium")
        self.momentum = neps.Float(
            lower=0.1, upper=0.999, default=0.4, default_confidence="low")
        self.weight_dec_active = neps.Categorical(
            choices=["True", "False"], default="False", default_confidence="low")
        self.weight_dec_adam = neps.Float(
            lower=0.00001, upper=0.1, default=1e-4, default_confidence="low")
        self.weight_dec_sgd = neps.Float(
            lower=0.00001, upper=0.1, default=1e-4, default_confidence="low")

        # Data augmentation
        self.random_horizontal_flip_prob = neps.Float(
            lower=0.0, upper=1.0, default=0.5, default_confidence="medium")
        self.random_vertical_flip_prob = neps.Float(
            lower=0.0, upper=1.0, default=0.5, default_confidence="low")
        self.random_rotation_deg = neps.Integer(
            lower=0, upper=180, default=90, default_confidence="low")
        self.random_rotation_prob = neps.Float(
            lower=0.0, upper=1.0, default=0.5, default_confidence="low")
        self.random_gaussian_noise_mean = neps.Float(
            lower=0.0, upper=1.0, default=0.0, default_confidence="medium")
        self.random_gaussian_noise_std = neps.Float(
            lower=0.0, upper=1.0, default=0.1, default_confidence="low")
        self.random_gaussian_noise_prob = neps.Float(
            lower=0.0, upper=1.0, default=0.5, default_confidence="low")
        self.brightness = neps.Float(
            lower=0.0, upper=1.0, default=0.5, default_confidence="low")
        self.contrast = neps.Float(
            lower=0.0, upper=1.0, default=0.5, default_confidence="low")
        self.saturation = neps.Float(
            lower=0.0, upper=1.0, default=0.5, default_confidence="low")

    def get_pipeline_space(self, pid, seed,  dataset, reduced_dataset_ratio):
        return {
            # Constant values
            "pid": neps.Constant(value=pid),
            "seed": neps.Constant(value=seed),
            "dataset": neps.Constant(value=dataset),
            "reduced_dataset_ratio": neps.Constant(value=reduced_dataset_ratio),

            # Hyperparameters
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,

            "scheduler": self.scheduler,

            "optimizer": self.optimizer,
            "momentum": self.momentum,
            "weight_dec_active": self.weight_dec_active,
            "weight_dec_adam": self.weight_dec_adam,
            "weight_dec_sgd": self.weight_dec_sgd,
            "random_horizontal_flip_prob": self.random_horizontal_flip_prob,
            "random_vertical_flip_prob": self.random_vertical_flip_prob,
            "random_rotation_deg": self.random_rotation_deg,
            "random_rotation_prob": self.random_rotation_prob,
            "random_gaussian_noise_mean": self.random_gaussian_noise_mean,
            "random_gaussian_noise_std": self.random_gaussian_noise_std,
            "random_gaussian_noise_prob": self.random_gaussian_noise_prob,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
        }
