import os

switch_to = os.environ.get("switch_to")
print(switch_to)

if switch_to == "modified":
    """CTGAN module."""

    import warnings

    import numpy as np
    import pandas as pd
    import torch
    from packaging import version
    from torch import optim
    from torch.nn import (
        BatchNorm1d,
        Dropout,
        LeakyReLU,
        Linear,
        Module,
        ReLU,
        Sequential,
        functional,
    )

    from ctgan.data_sampler import DataSampler
    from ctgan.data_transformer import DataTransformer
    from ctgan.synthesizers.base import BaseSynthesizer, random_state

    from sdmetrics.single_table.efficacy.binary import BinaryLogisticRegression
    from ctgan.experiments.logger import Logger, savefig
    from ctgan.experiments.utils import column_metric_wrapper, histogram_intersection
    from functools import partial
    import copy
    from ctgan.experiments.utils import data_transformer as customized_data_transformer

    import pdb
    from sklearn.decomposition import PCA

    class Discriminator(Module):
        """Discriminator for the CTGAN."""

        def __init__(self, input_dim, discriminator_dim, pac=10):
            super(Discriminator, self).__init__()
            dim = input_dim * pac
            self.pac = pac
            self.pacdim = dim
            seq = []
            for item in list(discriminator_dim):
                seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
                dim = item

            seq += [Linear(dim, 1)]
            self.seq = Sequential(*seq)

        def calc_gradient_penalty(
            self, real_data, fake_data, device="cpu", pac=10, lambda_=10
        ):
            """Compute the gradient penalty."""
            alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
            alpha = alpha.repeat(1, pac, real_data.size(1))
            alpha = alpha.view(-1, real_data.size(1))

            interpolates = alpha * real_data + ((1 - alpha) * fake_data)

            disc_interpolates = self(interpolates)

            gradients = torch.autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            gradients_view = (
                gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
            )
            gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

            return gradient_penalty

        def forward(self, input_):
            """Apply the Discriminator to the `input_`."""
            assert input_.size()[0] % self.pac == 0
            return self.seq(input_.view(-1, self.pacdim))

    class Residual(Module):
        """Residual layer for the CTGAN."""

        def __init__(self, i, o):
            super(Residual, self).__init__()
            self.fc = Linear(i, o)
            self.bn = BatchNorm1d(o)
            self.relu = ReLU()

        def forward(self, input_):
            """Apply the Residual layer to the `input_`."""
            out = self.fc(input_)
            out = self.bn(out)
            out = self.relu(out)
            return torch.cat([out, input_], dim=1)

    class Generator(Module):
        """Generator for the CTGAN."""

        def __init__(self, embedding_dim, generator_dim, data_dim):
            super(Generator, self).__init__()
            dim = embedding_dim
            seq = []
            for item in list(generator_dim):
                seq += [Residual(dim, item)]
                dim += item
            seq.append(Linear(dim, data_dim))
            self.seq = Sequential(*seq)

        def forward(self, input_):
            """Apply the Generator to the `input_`."""
            data = self.seq(input_)
            return data

    class CTGAN(BaseSynthesizer):
        """Conditional Table GAN Synthesizer.

        This is the core class of the CTGAN project, where the different components
        are orchestrated together.
        For more details about the process, please check the [Modeling Tabular data using
        Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

        Args:
            embedding_dim (int):
                Size of the random sample passed to the Generator. Defaults to 128.
            generator_dim (tuple or list of ints):
                Size of the output samples for each one of the Residuals. A Residual Layer
                will be created for each one of the values provided. Defaults to (256, 256).
            discriminator_dim (tuple or list of ints):
                Size of the output samples for each one of the Discriminator Layers. A Linear Layer
                will be created for each one of the values provided. Defaults to (256, 256).
            generator_lr (float):
                Learning rate for the generator. Defaults to 2e-4.
            generator_decay (float):
                Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
            discriminator_lr (float):
                Learning rate for the discriminator. Defaults to 2e-4.
            discriminator_decay (float):
                Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
            batch_size (int):
                Number of data samples to process in each step.
            discriminator_steps (int):
                Number of discriminator updates to do for each generator update.
                From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
                default is 5. Default used is 1 to match original CTGAN implementation.
            log_frequency (boolean):
                Whether to use log frequency of categorical levels in conditional
                sampling. Defaults to ``True``.
            verbose (boolean):
                Whether to have print statements for progress results. Defaults to ``False``.
            epochs (int):
                Number of training epochs. Defaults to 300.
            pac (int):
                Number of samples to group together when applying the discriminator.
                Defaults to 10.
            cuda (bool):
                Whether to attempt to use cuda for GPU computation.
                If this is False or CUDA is not available, CPU will be used.
                Defaults to ``True``.
            log_dir (str):
                Directory to save logs
        """

        def __init__(
            self,
            embedding_dim=128,
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            generator_lr=2e-4,
            generator_decay=1e-6,
            discriminator_lr=2e-4,
            discriminator_decay=1e-6,
            batch_size=500,
            discriminator_steps=1,
            log_frequency=True,
            verbose=False,
            epochs=300,
            pac=10,
            cuda=True,
            disable_condvec=False,
            log_dict=None,
            evaluation_dict=None,
            generator_penalty_dict=None,
        ):

            assert batch_size % 2 == 0

            self._embedding_dim = embedding_dim
            self._generator_dim = generator_dim
            self._discriminator_dim = discriminator_dim

            self._generator_lr = generator_lr
            self._generator_decay = generator_decay
            self._discriminator_lr = discriminator_lr
            self._discriminator_decay = discriminator_decay

            self._batch_size = batch_size
            self._discriminator_steps = discriminator_steps
            self._log_frequency = log_frequency
            self._verbose = verbose
            self._epochs = epochs
            self.pac = pac

            self._disable_condvec = disable_condvec
            self._log_dict = log_dict
            self._evaluation_dict = evaluation_dict
            self._generator_penalty_dict = generator_penalty_dict

            if not cuda or not torch.cuda.is_available():
                device = "cpu"
            elif isinstance(cuda, str):
                device = cuda
            else:
                device = "cuda"

            self._device = torch.device(device)

            self._transformer = None
            self._data_sampler = None
            self._generator = None
            

        @staticmethod
        def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
            """Deals with the instability of the gumbel_softmax for older versions of torch.

            For more details about the issue:
            https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

            Args:
                logits […, num_features]:
                    Unnormalized log probabilities
                tau:
                    Non-negative scalar temperature
                hard (bool):
                    If True, the returned samples will be discretized as one-hot vectors,
                    but will be differentiated as if it is the soft sample in autograd
                dim (int):
                    A dimension along which softmax will be computed. Default: -1.

            Returns:
                Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
            """
            if version.parse(torch.__version__) < version.parse("1.2.0"):
                for i in range(10):
                    transformed = functional.gumbel_softmax(
                        logits, tau=tau, hard=hard, eps=eps, dim=dim
                    )
                    if not torch.isnan(transformed).any():
                        return transformed
                raise ValueError("gumbel_softmax returning NaN.")

            return functional.gumbel_softmax(
                logits, tau=tau, hard=hard, eps=eps, dim=dim
            )

        def _apply_activate(self, data):
            """Apply proper activation function to the output of the generator."""
            data_t = []
            st = 0
            for column_info in self._transformer.output_info_list:
                for span_info in column_info:
                    if span_info.activation_fn == "tanh":
                        ed = st + span_info.dim
                        data_t.append(torch.tanh(data[:, st:ed]))
                        st = ed
                    elif span_info.activation_fn == "softmax":
                        ed = st + span_info.dim
                        transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                        data_t.append(transformed)
                        st = ed
                    else:
                        raise ValueError(
                            f"Unexpected activation function {span_info.activation_fn}."
                        )

            return torch.cat(data_t, dim=1)

        def _validate_discrete_columns(self, train_data, discrete_columns):
            """Check whether ``discrete_columns`` exists in ``train_data``.

            Args:
                train_data (numpy.ndarray or pandas.DataFrame):
                    Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
                discrete_columns (list-like):
                    List of discrete columns to be used to generate the Conditional
                    Vector. If ``train_data`` is a Numpy array, this list should
                    contain the integer indices of the columns. Otherwise, if it is
                    a ``pandas.DataFrame``, this list should contain the column names.
            """
            if isinstance(train_data, pd.DataFrame):
                invalid_columns = set(discrete_columns) - set(train_data.columns)
            elif isinstance(train_data, np.ndarray):
                invalid_columns = []
                for column in discrete_columns:
                    if column < 0 or column >= train_data.shape[1]:
                        invalid_columns.append(column)
            else:
                raise TypeError(
                    "``train_data`` should be either pd.DataFrame or np.array."
                )

            if invalid_columns:
                raise ValueError(f"Invalid columns found: {invalid_columns}")
            
        def _cond_loss(self, data, c, m):
            """Compute the cross entropy loss on the fixed discrete column."""
            loss = []
            st = 0
            st_c = 0
            for column_info in self._transformer.output_info_list:
                for span_info in column_info:
                    if len(column_info) != 1 or span_info.activation_fn != "softmax":
                        # not discrete column
                        st += span_info.dim
                    else:
                        ed = st + span_info.dim
                        ed_c = st_c + span_info.dim
                        tmp = functional.cross_entropy(
                            data[:, st:ed],
                            torch.argmax(c[:, st_c:ed_c], dim=1),
                            reduction="none",
                        )
                        loss.append(tmp)
                        st = ed
                        st_c = ed_c

            loss = torch.stack(loss, dim=1)  # noqa: PD013

            return (loss * m).sum() / data.size()[0]

        def _info_loss_sdgym(self, data_real, data_fake, norm=1):
            """Tablegan stdgym info loss implementation.

            Args:
                data_real (torch.Tensor):
                    Real data (b, m)
                data_fake (torch.Tensor):
                    Fake data with activation applied (b, m).
                norm (int, optional):
                    The order of norm. Defaults to 1.

            Returns:
                loss_info:
                    Summation of the norm of expected mean and
                    standard deviation difference between the real
                    and fake data.
            """
            loss_mean = torch.norm(
                torch.mean(data_fake, dim=0) - torch.mean(data_real, dim=0), p=norm
            )
            loss_std = torch.norm(
                torch.std(data_fake, dim=0) - torch.std(data_real, dim=0), p=norm
            )
            loss_info = loss_mean + loss_std

            return loss_info

        def _info_loss_tablegan(
            self, moving_average_type, id, data_real, data_fake
        ):
            model_disc_copy = copy.deepcopy(self._discriminator)
            model_disc_copy.seq = Sequential(*list(self._discriminator.seq.children())[:-1])
            real_features = model_disc_copy(data_real)
            fake_features = model_disc_copy(data_fake)
            if moving_average_type == "simple":
                (
                    real_mean_features,
                    fake_mean_features,
                    real_std_features,
                    fake_std_features,
                ) = self._simple_moving_average(id, real_features, fake_features)

            elif moving_average_type == "exponential":
                (
                    real_mean_features,
                    fake_mean_features,
                    real_std_features,
                    fake_std_features,
                ) = self._exponential_moving_average(id, real_features, fake_features)
            else:
                real_mean_features = real_features.mean(dim=0)  
                fake_mean_features = fake_features.mean(dim=0)
                real_std_features = real_features.std(dim=0)
                fake_std_features = fake_features.std(dim=0)

            loss_mean = torch.norm(real_mean_features - fake_mean_features, p=2)
            loss_std = torch.norm(real_std_features - fake_std_features, p=2)

            loss_info = loss_mean + loss_std
            return loss_info

        def _simple_moving_average(self, id, real_features, fake_features):
            real_mean_features = real_features.mean(dim=0)
            fake_mean_features = fake_features.mean(dim=0)
            real_std_features = real_features.std(dim=0)
            fake_std_features = fake_features.std(dim=0)

            if id == 0:
                self._prev_real_mean_features = real_mean_features.detach().view(1, -1)
                self._prev_fake_mean_features = fake_mean_features.detach().view(1, -1)
                self._prev_real_std_features = real_std_features.detach().view(1, -1)
                self._prev_fake_std_features = fake_std_features.detach().view(1, -1)

            else:
                self._prev_real_mean_features = torch.cat(
                    [
                        self._prev_real_mean_features,
                        real_mean_features.detach().view(1, -1),
                    ]
                )
                self._prev_fake_mean_features = torch.cat(
                    [
                        self._prev_fake_mean_features,
                        fake_mean_features.detach().view(1, -1),
                    ]
                )
                self._prev_real_std_features = torch.cat(
                    [
                        self._prev_real_std_features,
                        real_std_features.detach().view(1, -1),
                    ]
                )
                self._prev_fake_std_features = torch.cat(
                    [
                        self._prev_fake_std_features,
                        real_std_features.detach().view(1, -1),
                    ]
                )

                real_mean_features = self._prev_real_mean_features.mean(dim=0)
                fake_mean_features = self._prev_fake_mean_features.mean(dim=0)
                real_std_features = self._prev_real_std_features.std(dim=0)
                fake_std_features = self._prev_fake_std_features.std(dim=0)

                # Enable gradient computation
                real_mean_features.requires_grad = True
                fake_mean_features.requires_grad = True
                real_std_features.requires_grad = True
                fake_std_features.requires_grad = True

            return (
                real_mean_features,
                fake_mean_features,
                real_std_features,
                fake_std_features,
            )

        def _exponential_moving_average(self, id, real_features, fake_features):
            real_mean_features = real_features.mean(dim=0)
            fake_mean_features = fake_features.mean(dim=0)
            real_std_features = real_features.std(dim=0)
            fake_std_features = fake_features.std(dim=0)

            smoothing_factor = 2 / (1 + self._steps_per_epoch)

            def _moving_average_update(curr_val, previousEMA, smoothing_factor):
                current_EMA = (smoothing_factor * curr_val) + (
                    (1 - smoothing_factor) * previousEMA
                )
                return current_EMA

            if id == 0:
                self._prev_real_mean_features = torch.zeros_like(
                    real_mean_features, requires_grad=False
                )
                self._prev_fake_mean_features = torch.zeros_like(
                    real_mean_features, requires_grad=False
                )
                self._prev_real_std_features = torch.zeros_like(
                    real_mean_features, requires_grad=False
                )
                self._prev_fake_std_features = torch.zeros_like(
                    real_mean_features, requires_grad=False
                )

            real_mean_features = _moving_average_update(
                curr_val=real_mean_features,
                previousEMA=self._prev_real_mean_features,
                smoothing_factor=smoothing_factor,
            )
            fake_mean_features = _moving_average_update(
                curr_val=fake_mean_features,
                previousEMA=self._prev_fake_mean_features,
                smoothing_factor=smoothing_factor,
            )
            real_std_features = _moving_average_update(
                curr_val=real_std_features,
                previousEMA=self._prev_real_std_features,
                smoothing_factor=smoothing_factor,
            )

            fake_std_features = _moving_average_update(
                curr_val=fake_std_features,
                previousEMA=self._prev_fake_std_features,
                smoothing_factor=smoothing_factor,
            )

            # Detach gradient computation
            self._prev_real_mean_features = real_mean_features.detach()
            self._prev_fake_mean_features = fake_mean_features.detach()
            self._prev_real_std_features = real_std_features.detach()
            self._prev_fake_std_features = fake_std_features.detach()

            return (
                real_mean_features,
                fake_mean_features,
                real_std_features,
                fake_std_features,
            )

        def _pca_loss(self, data_real, data_fake):
            """PCA-embedded data loss implementation.

            Args:
                data_real (torch.Tensor):
                    Real data (b, m)
                data_fake (torch.Tensor):
                    Fake data with activation applied (b, m).

            Returns:
                loss_info:
                    Summation of the norm of expected mean and
                    standard deviation difference between the real
                    and fake data.
            """
            
            data_real = torch.from_numpy(self._pca.transform(data_real.detach().numpy()))
            data_fake = torch.from_numpy(self._pca.transform(data_fake.detach().numpy()))
            
            data_real.requires_grad = True
            data_fake.requires_grad = True
            
            loss_mean = torch.norm(
                torch.mean(data_fake, dim=0) - torch.mean(data_real, dim=0), p=2
            )
            loss_std = torch.norm(
                torch.std(data_fake, dim=0) - torch.std(data_real, dim=0), p=2
            )
            loss_info = loss_mean + loss_std

            return loss_info

        @random_state
        def fit(self, train_data, discrete_columns=(), epochs=None):
            """Fit the CTGAN Synthesizer models to the training data.

            Args:
                train_data (numpy.ndarray or pandas.DataFrame):
                    Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
                discrete_columns (list-like):
                    List of discrete columns to be used to generate the Conditional
                    Vector. If ``train_data`` is a Numpy array, this list should
                    contain the integer indices of the columns. Otherwise, if it is
                    a ``pandas.DataFrame``, this list should contain the column names.
            """
            self._validate_discrete_columns(train_data, discrete_columns)

            if epochs is None:
                epochs = self._epochs
            else:
                warnings.warn(
                    (
                        "`epochs` argument in `fit` method has been deprecated and will be removed "
                        "in a future version. Please pass `epochs` to the constructor instead"
                    ),
                    DeprecationWarning,
                )

            ### Train data evaluation
            eval_data = train_data.copy()
            target = self._evaluation_dict["target"]
            ###

            self._transformer = DataTransformer()
            self._transformer.fit(train_data, discrete_columns)

            train_data = self._transformer.transform(train_data)

            self._data_sampler = DataSampler(
                train_data, self._transformer.output_info_list, self._log_frequency
            )

            data_dim = self._transformer.output_dimensions
            
            if "pca_loss" in self._generator_penalty_dict["loss"]:
                # fit pca model on the train_data
                n_components = min(train_data.shape[0], train_data.shape[1])
                n_components = min(n_components, 100)
                self._pca = PCA(n_components=n_components)
                self._pca.fit(train_data)

            if self._disable_condvec:
                self._generator = Generator(
                    self._embedding_dim,
                    self._generator_dim,
                    data_dim,
                ).to(self._device)

                self._discriminator = Discriminator(
                    data_dim,
                    self._discriminator_dim,
                    pac=self.pac,
                ).to(self._device)
            else:
                self._generator = Generator(
                    self._embedding_dim + self._data_sampler.dim_cond_vec(),
                    self._generator_dim,
                    data_dim,
                ).to(self._device)
                
                self._discriminator = Discriminator(
                    data_dim + self._data_sampler.dim_cond_vec(),
                    self._discriminator_dim,
                    pac=self.pac,
                ).to(self._device)

            optimizerG = optim.Adam(
                self._generator.parameters(),
                lr=self._generator_lr,
                betas=(0.5, 0.9),
                weight_decay=self._generator_decay,
            )

            optimizerD = optim.Adam(
                self._discriminator.parameters(),
                lr=self._discriminator_lr,
                betas=(0.5, 0.9),
                weight_decay=self._discriminator_decay,
            )

            mean = torch.zeros(
                self._batch_size, self._embedding_dim, device=self._device
            )
            std = mean + 1

            self._steps_per_epoch = max(len(train_data) // self._batch_size, 1)

            ### Setup Logger
            title = self._log_dict["title"]
            log_dir = self._log_dict["log_dir"]
            plot_title = self._log_dict["plot_title"]
            logger = Logger(f"{log_dir}/log.txt", title=title)
            logger.set_names(
                [
                    "Epoch",
                    "F1/fake",
                    "F1/real",
                    "Hist/real",
                    "Hist/fake",
                    "Loss/disc",
                    "Loss/gen",
                    "Loss/gen/adv",
                    "Loss/gen/cond",
                    "Loss/gen/sdgym_info",
                    "Loss/gen/tablegan_info",
                    "Loss/gen/pca",
                ]
            )
            ###

            for i in range(epochs):
                ### Create empty list
                log = []
                ###
                # self._generator.train()
                for id_ in range(self._steps_per_epoch):

                    for n in range(self._discriminator_steps):
                        fakez = torch.normal(mean=mean, std=std)

                        if self._disable_condvec:
                            condvec = None
                        else:
                            condvec = self._data_sampler.sample_condvec(
                                self._batch_size
                            )

                        if condvec is None:
                            c1, m1, col, opt = None, None, None, None
                            real = self._data_sampler.sample_data(
                                self._batch_size, col, opt
                            )
                        else:
                            c1, m1, col, opt = condvec
                            c1 = torch.from_numpy(c1).to(self._device)
                            m1 = torch.from_numpy(m1).to(self._device)
                            fakez = torch.cat([fakez, c1], dim=1)

                            perm = np.arange(self._batch_size)
                            np.random.shuffle(perm)
                            real = self._data_sampler.sample_data(
                                self._batch_size, col[perm], opt[perm]
                            )
                            c2 = c1[perm]

                        fake = self._generator(fakez)
                        fakeact = self._apply_activate(fake)

                        real = torch.from_numpy(real.astype("float32")).to(self._device)

                        if c1 is not None:
                            fake_cat = torch.cat([fakeact, c1], dim=1)
                            real_cat = torch.cat([real, c2], dim=1)
                        else:
                            real_cat = real
                            fake_cat = fakeact

                        y_fake = self._discriminator(fake_cat)
                        y_real = self._discriminator(real_cat)

                        pen = self._discriminator.calc_gradient_penalty(
                            real_cat, fake_cat, self._device, self.pac
                        )
                        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                        optimizerD.zero_grad()
                        pen.backward(retain_graph=True)
                        loss_d.backward()
                        optimizerD.step()

                    fakez = torch.normal(mean=mean, std=std)

                    if self._disable_condvec:
                        condvec = None
                    else:
                        condvec = self._data_sampler.sample_condvec(self._batch_size)

                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if c1 is not None:
                        y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                    else:
                        y_fake = self._discriminator(fakeact)

                    # Initialize the different types of losses defined
                    loss_g_adv = torch.Tensor([0])
                    loss_g_cond = torch.Tensor([0])
                    loss_g_sdgym_info = torch.Tensor([0])
                    loss_g_tablegan_info = torch.Tensor([0])
                    loss_g_pca = torch.Tensor([0])

                    # Clear accumulated gradient
                    optimizerG.zero_grad()

                    if self._generator_penalty_dict is not None:
                        losses = self._generator_penalty_dict["loss"]

                        for loss in losses:
                            if condvec is None:
                                c1, m1, col, opt = None, None, None, None
                                real = self._data_sampler.sample_data(
                                    self._batch_size, col, opt
                                )
                                real = torch.from_numpy(real.astype("float32")).to(
                                    self._device
                                )
                                real_cat = real
                                fake_cat = fakeact
                            else:
                                perm = np.arange(self._batch_size)
                                np.random.shuffle(perm)
                                real = self._data_sampler.sample_data(
                                    self._batch_size, col[perm], opt[perm]
                                )
                                real = torch.from_numpy(real.astype("float32")).to(
                                    self._device
                                )
                                c2 = c1[perm]
                                real_cat = torch.cat([real, c2], dim=1)
                                fake_cat = torch.cat([fakeact, c1], dim=1)

                            if loss == "wass_gan":
                                loss_g_adv = -torch.mean(y_fake)
                            elif loss == "cond":
                                if condvec is None:
                                    print(
                                        f"Condition vector is None! Cannot compute loss."
                                    )
                                    loss_g_cond = torch.Tensor([0])
                                else:
                                    loss_g_cond = self._cond_loss(fake, c1, m1)
                            elif loss == "l2_sdgym_info":
                                loss_g_sdgym_info = self._info_loss_sdgym(
                                    data_real=real, data_fake=fakeact, norm=2
                                )
                            elif loss == "l1_sdgym_info":
                                loss_g_sdgym_info = self._info_loss_sdgym(
                                    data_real=real, data_fake=fakeact, norm=1
                                )
                            elif loss == "tablegan_info":
                                loss_g_tablegan_info = self._info_loss_tablegan(
                                    moving_average_type=None,
                                    id=id_,
                                    data_real=real_cat,
                                    data_fake=fake_cat,
                                )
                            elif loss == "minimize_tablegan_info":
                                loss_g_tablegan_info = self._info_loss_tablegan(
                                    moving_average_type=None,
                                    id=id_,
                                    data_real=real_cat,
                                    data_fake=fake_cat,
                                )
                                loss_g_tablegan_info = - loss_g_tablegan_info
                            elif loss == "exp_tablegan_info":
                                loss_g_tablegan_info = self._info_loss_tablegan(
                                    moving_average_type="exponential",
                                    id=id_,
                                    data_real=real_cat,
                                    data_fake=fake_cat,
                                )
                            elif loss == "simple_tablegan_info":
                                fake_cat = torch.cat([fakeact, c1], dim=1)
                                real_cat = torch.cat([real, c2], dim=1)
                                loss_g_tablegan_info = self._info_loss_tablegan(
                                    moving_average_type="simple",
                                    id=id_,
                                    data_real=real_cat,
                                    data_fake=fake_cat,
                                )
                            elif loss == "pca_loss":
                                loss_g_pca = self._pca_loss(
                                    data_real=real, data_fake=fakeact
                                )
                            else:
                                raise NotImplementedError(
                                    f"`{loss}` is not implemented"
                                )

                        loss_g = (
                            loss_g_adv
                            + loss_g_cond
                            + loss_g_sdgym_info
                            + loss_g_tablegan_info
                            + loss_g_pca
                        )
                        loss_g.backward()
                    else:
                        loss_g_adv = -torch.mean(y_fake)
                        if condvec is None:
                            loss_g_cond = torch.Tensor([0])
                        else:
                            loss_g_cond = self._cond_loss(fake, c1, m1)

                        loss_g = loss_g_adv + loss_g_cond
                        loss_g.backward()

                    # Update the gradients connected to the generator
                    optimizerG.step()

                if self._verbose:
                    ### Evaluate fake data
                    fakedata = self.sample(n=self._batch_size)
                    realdata = self._transformer.inverse_transform(real.numpy())
                    assert fakedata.shape[1] == realdata.shape[1] == eval_data.shape[1]

                    (
                        transformed_testdata,
                        transformed_traindata,
                        _,
                    ) = customized_data_transformer(
                        realdata=eval_data, fakedata=realdata
                    )
                    real_f1 = BinaryLogisticRegression.compute(
                        test_data=transformed_testdata,
                        train_data=transformed_traindata,
                        target=target,
                    )
                    (
                        transformed_testdata,
                        transformed_fakedata,
                        _,
                    ) = customized_data_transformer(
                        realdata=eval_data, fakedata=fakedata
                    )
                    fake_f1 = BinaryLogisticRegression.compute(
                        test_data=transformed_testdata,
                        train_data=transformed_fakedata,
                        target=target,
                    )

                    func = partial(
                        column_metric_wrapper,
                        column_metric=partial(histogram_intersection, bins=50),
                        cat_cols=discrete_columns,
                    )
                    fake_hist = func(realdata=eval_data, fakedata=fakedata).score.mean()
                    real_hist = func(
                        realdata=eval_data,
                        fakedata=realdata,
                    ).score.mean()

                    print(
                        f"Epoch {i+1}, Loss G: {loss_g.item(): .4f},"  # noqa: T001
                        f"Loss D: {loss_d.item(): .4f}",
                        f"F1: {fake_f1}",
                        f"Hist: {fake_hist}",
                        flush=True,
                    )

                    ### Update log and plot result
                    log.extend(
                        [
                            i + 1,
                            fake_f1,
                            real_f1,
                            real_hist,
                            fake_hist,
                            loss_d.item(),
                            loss_g.item(),
                            loss_g_adv.item(),
                            loss_g_cond.item(),
                            loss_g_sdgym_info.item(),
                            loss_g_tablegan_info.item(),
                            loss_g_pca.item(),
                        ]
                    )
                    logger.append(log)
                    logger.plot(
                        ["Loss/disc", "Loss/gen", "Loss/gen/adv"],
                        x=None,
                        xlabel="Epoch",
                        title=plot_title,
                    )
                    savefig(f"{log_dir}/loss.png")
                    logger.plot(
                        [
                            "Loss/gen/cond",
                            "Loss/gen/sdgym_info",
                            "Loss/gen/tablegan_info",
                            "Loss/gen/pca",
                        ],
                        x=None,
                        xlabel="Epoch",
                        title=plot_title,
                    )
                    savefig(f"{log_dir}/loss_gen_pen.png")
                    logger.plot(
                        [
                            "F1/real",
                            "F1/fake",
                            "Hist/real",
                            "Hist/fake",
                        ],
                        x=None,
                        xlabel="Epoch",
                        title=plot_title,
                    )
                    savefig(f"{log_dir}/metrics.png")
                    self.save(f"{log_dir}/model.pt")
                    ###

        @random_state
        def sample(self, n, condition_column=None, condition_value=None):
            """Sample data similar to the training data.

            Choosing a condition_column and condition_value will increase the probability of the
            discrete condition_value happening in the condition_column.

            Args:
                n (int):
                    Number of rows to sample.
                condition_column (string):
                    Name of a discrete column.
                condition_value (string):
                    Name of the category in the condition_column which we wish to increase the
                    probability of happening.

            Returns:
                numpy.ndarray or pandas.DataFrame
            """
            self._generator.eval()
            if condition_column is not None and condition_value is not None:
                condition_info = self._transformer.convert_column_name_value_to_id(
                    condition_column, condition_value
                )
                global_condition_vec = (
                    self._data_sampler.generate_cond_from_condition_column_info(
                        condition_info, self._batch_size
                    )
                )
            else:
                global_condition_vec = None

            steps = n // self._batch_size + 1
            data = []
            for i in range(steps):
                mean = torch.zeros(self._batch_size, self._embedding_dim)
                std = mean + 1
                fakez = torch.normal(mean=mean, std=std).to(self._device)

                if global_condition_vec is not None:
                    condvec = global_condition_vec.copy()
                else:
                    if self._disable_condvec:
                        condvec = None
                    else:
                        condvec = self._data_sampler.sample_original_condvec(
                            self._batch_size
                        )

                if condvec is None:
                    pass
                else:
                    c1 = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                data.append(fakeact.detach().cpu().numpy())

            data = np.concatenate(data, axis=0)
            data = data[:n]
            self._generator.train()
            return self._transformer.inverse_transform(data)

        def set_device(self, device):
            """Set the `device` to be used ('GPU' or 'CPU)."""
            self._device = device
            if self._generator is not None:
                self._generator.to(self._device)

else:
    """CTGAN module."""

    import warnings

    import numpy as np
    import pandas as pd
    import torch
    from packaging import version
    from torch import optim
    from torch.nn import (
        BatchNorm1d,
        Dropout,
        LeakyReLU,
        Linear,
        Module,
        ReLU,
        Sequential,
        functional,
    )

    from ctgan.data_sampler import DataSampler
    from ctgan.data_transformer import DataTransformer
    from ctgan.synthesizers.base import BaseSynthesizer, random_state

    class Discriminator(Module):
        """Discriminator for the CTGAN."""

        def __init__(self, input_dim, discriminator_dim, pac=10):
            super(Discriminator, self).__init__()
            dim = input_dim * pac
            self.pac = pac
            self.pacdim = dim
            seq = []
            for item in list(discriminator_dim):
                seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
                dim = item

            seq += [Linear(dim, 1)]
            self.seq = Sequential(*seq)

        def calc_gradient_penalty(
            self, real_data, fake_data, device="cpu", pac=10, lambda_=10
        ):
            """Compute the gradient penalty."""
            alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
            alpha = alpha.repeat(1, pac, real_data.size(1))
            alpha = alpha.view(-1, real_data.size(1))

            interpolates = alpha * real_data + ((1 - alpha) * fake_data)

            disc_interpolates = self(interpolates)

            gradients = torch.autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            gradients_view = (
                gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
            )
            gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

            return gradient_penalty

        def forward(self, input_):
            """Apply the Discriminator to the `input_`."""
            assert input_.size()[0] % self.pac == 0
            return self.seq(input_.view(-1, self.pacdim))

    class Residual(Module):
        """Residual layer for the CTGAN."""

        def __init__(self, i, o):
            super(Residual, self).__init__()
            self.fc = Linear(i, o)
            self.bn = BatchNorm1d(o)
            self.relu = ReLU()

        def forward(self, input_):
            """Apply the Residual layer to the `input_`."""
            out = self.fc(input_)
            out = self.bn(out)
            out = self.relu(out)
            return torch.cat([out, input_], dim=1)

    class Generator(Module):
        """Generator for the CTGAN."""

        def __init__(self, embedding_dim, generator_dim, data_dim):
            super(Generator, self).__init__()
            dim = embedding_dim
            seq = []
            for item in list(generator_dim):
                seq += [Residual(dim, item)]
                dim += item
            seq.append(Linear(dim, data_dim))
            self.seq = Sequential(*seq)

        def forward(self, input_):
            """Apply the Generator to the `input_`."""
            data = self.seq(input_)
            return data

    class CTGAN(BaseSynthesizer):
        """Conditional Table GAN Synthesizer.
        This is the core class of the CTGAN project, where the different components
        are orchestrated together.
        For more details about the process, please check the [Modeling Tabular data using
        Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
        Args:
            embedding_dim (int):
                Size of the random sample passed to the Generator. Defaults to 128.
            generator_dim (tuple or list of ints):
                Size of the output samples for each one of the Residuals. A Residual Layer
                will be created for each one of the values provided. Defaults to (256, 256).
            discriminator_dim (tuple or list of ints):
                Size of the output samples for each one of the Discriminator Layers. A Linear Layer
                will be created for each one of the values provided. Defaults to (256, 256).
            generator_lr (float):
                Learning rate for the generator. Defaults to 2e-4.
            generator_decay (float):
                Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
            discriminator_lr (float):
                Learning rate for the discriminator. Defaults to 2e-4.
            discriminator_decay (float):
                Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
            batch_size (int):
                Number of data samples to process in each step.
            discriminator_steps (int):
                Number of discriminator updates to do for each generator update.
                From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
                default is 5. Default used is 1 to match original CTGAN implementation.
            log_frequency (boolean):
                Whether to use log frequency of categorical levels in conditional
                sampling. Defaults to ``True``.
            verbose (boolean):
                Whether to have print statements for progress results. Defaults to ``False``.
            epochs (int):
                Number of training epochs. Defaults to 300.
            pac (int):
                Number of samples to group together when applying the discriminator.
                Defaults to 10.
            cuda (bool):
                Whether to attempt to use cuda for GPU computation.
                If this is False or CUDA is not available, CPU will be used.
                Defaults to ``True``.
        """

        def __init__(
            self,
            embedding_dim=128,
            generator_dim=(256, 256),
            discriminator_dim=(256, 256),
            generator_lr=2e-4,
            generator_decay=1e-6,
            discriminator_lr=2e-4,
            discriminator_decay=1e-6,
            batch_size=500,
            discriminator_steps=1,
            log_frequency=True,
            verbose=False,
            epochs=300,
            pac=10,
            cuda=True,
        ):

            assert batch_size % 2 == 0

            self._embedding_dim = embedding_dim
            self._generator_dim = generator_dim
            self._discriminator_dim = discriminator_dim

            self._generator_lr = generator_lr
            self._generator_decay = generator_decay
            self._discriminator_lr = discriminator_lr
            self._discriminator_decay = discriminator_decay

            self._batch_size = batch_size
            self._discriminator_steps = discriminator_steps
            self._log_frequency = log_frequency
            self._verbose = verbose
            self._epochs = epochs
            self.pac = pac

            if not cuda or not torch.cuda.is_available():
                device = "cpu"
            elif isinstance(cuda, str):
                device = cuda
            else:
                device = "cuda"

            self._device = torch.device(device)

            self._transformer = None
            self._data_sampler = None
            self._generator = None

        @staticmethod
        def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
            """Deals with the instability of the gumbel_softmax for older versions of torch.
            For more details about the issue:
            https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
            Args:
                logits […, num_features]:
                    Unnormalized log probabilities
                tau:
                    Non-negative scalar temperature
                hard (bool):
                    If True, the returned samples will be discretized as one-hot vectors,
                    but will be differentiated as if it is the soft sample in autograd
                dim (int):
                    A dimension along which softmax will be computed. Default: -1.
            Returns:
                Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
            """
            if version.parse(torch.__version__) < version.parse("1.2.0"):
                for i in range(10):
                    transformed = functional.gumbel_softmax(
                        logits, tau=tau, hard=hard, eps=eps, dim=dim
                    )
                    if not torch.isnan(transformed).any():
                        return transformed
                raise ValueError("gumbel_softmax returning NaN.")

            return functional.gumbel_softmax(
                logits, tau=tau, hard=hard, eps=eps, dim=dim
            )

        def _apply_activate(self, data):
            """Apply proper activation function to the output of the generator."""
            data_t = []
            st = 0
            for column_info in self._transformer.output_info_list:
                for span_info in column_info:
                    if span_info.activation_fn == "tanh":
                        ed = st + span_info.dim
                        data_t.append(torch.tanh(data[:, st:ed]))
                        st = ed
                    elif span_info.activation_fn == "softmax":
                        ed = st + span_info.dim
                        transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                        data_t.append(transformed)
                        st = ed
                    else:
                        raise ValueError(
                            f"Unexpected activation function {span_info.activation_fn}."
                        )

            return torch.cat(data_t, dim=1)

        def _cond_loss(self, data, c, m):
            """Compute the cross entropy loss on the fixed discrete column."""
            loss = []
            st = 0
            st_c = 0
            for column_info in self._transformer.output_info_list:
                for span_info in column_info:
                    if len(column_info) != 1 or span_info.activation_fn != "softmax":
                        # not discrete column
                        st += span_info.dim
                    else:
                        ed = st + span_info.dim
                        ed_c = st_c + span_info.dim
                        tmp = functional.cross_entropy(
                            data[:, st:ed],
                            torch.argmax(c[:, st_c:ed_c], dim=1),
                            reduction="none",
                        )
                        loss.append(tmp)
                        st = ed
                        st_c = ed_c

            loss = torch.stack(loss, dim=1)  # noqa: PD013

            return (loss * m).sum() / data.size()[0]

        def _validate_discrete_columns(self, train_data, discrete_columns):
            """Check whether ``discrete_columns`` exists in ``train_data``.
            Args:
                train_data (numpy.ndarray or pandas.DataFrame):
                    Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
                discrete_columns (list-like):
                    List of discrete columns to be used to generate the Conditional
                    Vector. If ``train_data`` is a Numpy array, this list should
                    contain the integer indices of the columns. Otherwise, if it is
                    a ``pandas.DataFrame``, this list should contain the column names.
            """
            if isinstance(train_data, pd.DataFrame):
                invalid_columns = set(discrete_columns) - set(train_data.columns)
            elif isinstance(train_data, np.ndarray):
                invalid_columns = []
                for column in discrete_columns:
                    if column < 0 or column >= train_data.shape[1]:
                        invalid_columns.append(column)
            else:
                raise TypeError(
                    "``train_data`` should be either pd.DataFrame or np.array."
                )

            if invalid_columns:
                raise ValueError(f"Invalid columns found: {invalid_columns}")

        @random_state
        def fit(self, train_data, discrete_columns=(), epochs=None):
            """Fit the CTGAN Synthesizer models to the training data.
            Args:
                train_data (numpy.ndarray or pandas.DataFrame):
                    Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
                discrete_columns (list-like):
                    List of discrete columns to be used to generate the Conditional
                    Vector. If ``train_data`` is a Numpy array, this list should
                    contain the integer indices of the columns. Otherwise, if it is
                    a ``pandas.DataFrame``, this list should contain the column names.
            """
            self._validate_discrete_columns(train_data, discrete_columns)

            if epochs is None:
                epochs = self._epochs
            else:
                warnings.warn(
                    (
                        "`epochs` argument in `fit` method has been deprecated and will be removed "
                        "in a future version. Please pass `epochs` to the constructor instead"
                    ),
                    DeprecationWarning,
                )

            self._transformer = DataTransformer()
            self._transformer.fit(train_data, discrete_columns)

            train_data = self._transformer.transform(train_data)

            self._data_sampler = DataSampler(
                train_data, self._transformer.output_info_list, self._log_frequency
            )

            data_dim = self._transformer.output_dimensions

            self._generator = Generator(
                self._embedding_dim + self._data_sampler.dim_cond_vec(),
                self._generator_dim,
                data_dim,
            ).to(self._device)

            discriminator = Discriminator(
                data_dim + self._data_sampler.dim_cond_vec(),
                self._discriminator_dim,
                pac=self.pac,
            ).to(self._device)

            optimizerG = optim.Adam(
                self._generator.parameters(),
                lr=self._generator_lr,
                betas=(0.5, 0.9),
                weight_decay=self._generator_decay,
            )

            optimizerD = optim.Adam(
                discriminator.parameters(),
                lr=self._discriminator_lr,
                betas=(0.5, 0.9),
                weight_decay=self._discriminator_decay,
            )

            mean = torch.zeros(
                self._batch_size, self._embedding_dim, device=self._device
            )
            std = mean + 1

            steps_per_epoch = max(len(train_data) // self._batch_size, 1)
            for i in range(epochs):
                for id_ in range(steps_per_epoch):

                    for n in range(self._discriminator_steps):
                        fakez = torch.normal(mean=mean, std=std)

                        condvec = self._data_sampler.sample_condvec(self._batch_size)
                        if condvec is None:
                            c1, m1, col, opt = None, None, None, None
                            real = self._data_sampler.sample_data(
                                self._batch_size, col, opt
                            )
                        else:
                            c1, m1, col, opt = condvec
                            c1 = torch.from_numpy(c1).to(self._device)
                            m1 = torch.from_numpy(m1).to(self._device)
                            fakez = torch.cat([fakez, c1], dim=1)

                            perm = np.arange(self._batch_size)
                            np.random.shuffle(perm)
                            real = self._data_sampler.sample_data(
                                self._batch_size, col[perm], opt[perm]
                            )
                            c2 = c1[perm]

                        fake = self._generator(fakez)
                        fakeact = self._apply_activate(fake)

                        real = torch.from_numpy(real.astype("float32")).to(self._device)

                        if c1 is not None:
                            fake_cat = torch.cat([fakeact, c1], dim=1)
                            real_cat = torch.cat([real, c2], dim=1)
                        else:
                            real_cat = real
                            fake_cat = fakeact

                        y_fake = discriminator(fake_cat)
                        y_real = discriminator(real_cat)

                        pen = discriminator.calc_gradient_penalty(
                            real_cat, fake_cat, self._device, self.pac
                        )
                        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                        optimizerD.zero_grad()
                        pen.backward(retain_graph=True)
                        loss_d.backward()
                        optimizerD.step()

                    fakez = torch.normal(mean=mean, std=std)
                    condvec = self._data_sampler.sample_condvec(self._batch_size)

                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if c1 is not None:
                        y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                    else:
                        y_fake = discriminator(fakeact)

                    if condvec is None:
                        cross_entropy = 0
                    else:
                        cross_entropy = self._cond_loss(fake, c1, m1)

                    loss_g = -torch.mean(y_fake) + cross_entropy

                    optimizerG.zero_grad()
                    loss_g.backward()
                    optimizerG.step()

                if self._verbose:
                    print(
                        f"Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},"  # noqa: T001
                        f"Loss D: {loss_d.detach().cpu(): .4f}",
                        flush=True,
                    )

        @random_state
        def sample(self, n, condition_column=None, condition_value=None):
            """Sample data similar to the training data.
            Choosing a condition_column and condition_value will increase the probability of the
            discrete condition_value happening in the condition_column.
            Args:
                n (int):
                    Number of rows to sample.
                condition_column (string):
                    Name of a discrete column.
                condition_value (string):
                    Name of the category in the condition_column which we wish to increase the
                    probability of happening.
            Returns:
                numpy.ndarray or pandas.DataFrame
            """
            if condition_column is not None and condition_value is not None:
                condition_info = self._transformer.convert_column_name_value_to_id(
                    condition_column, condition_value
                )
                global_condition_vec = (
                    self._data_sampler.generate_cond_from_condition_column_info(
                        condition_info, self._batch_size
                    )
                )
            else:
                global_condition_vec = None

            steps = n // self._batch_size + 1
            data = []
            for i in range(steps):
                mean = torch.zeros(self._batch_size, self._embedding_dim)
                std = mean + 1
                fakez = torch.normal(mean=mean, std=std).to(self._device)

                if global_condition_vec is not None:
                    condvec = global_condition_vec.copy()
                else:
                    condvec = self._data_sampler.sample_original_condvec(
                        self._batch_size
                    )

                if condvec is None:
                    pass
                else:
                    c1 = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                data.append(fakeact.detach().cpu().numpy())

            data = np.concatenate(data, axis=0)
            data = data[:n]

            return self._transformer.inverse_transform(data)

        def set_device(self, device):
            """Set the `device` to be used ('GPU' or 'CPU)."""
            self._device = device
            if self._generator is not None:
                self._generator.to(self._device)
