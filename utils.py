import jax
import matplotlib.pyplot as plt
import ipywidgets
import numpy as np
from IPython.display import clear_output
import jax.numpy as jnp


class Plotter:
    def __init__(self, params, forward, loss, dataset, output):

        self.params = params
        self.forward = forward
        self.dataset = dataset

        self.loss_fn = loss
        self.grad_fn = jax.grad(self.loss_fn)

        self.vx = 1.25 * max(abs(dataset.x))
        self.vy = 1.25 * max(abs(dataset.y))

        self.output = output

        self.mode = 'Full dataset'
        self.title = ''

        self.update_params_on_change = True

    def display(self):
        with self.output:
            clear_output(True)
            plt.xlim(-self.vx, self.vx)
            plt.ylim(-self.vy, self.vy)

            if self.mode == 'Full dataset':
                plt.scatter(self.dataset.x, self.dataset.y, color='#DA702C')

            elif self.mode == 'Subsampled dataset':
                plt.scatter(self.dataset.xsmp, self.dataset.ysmp, color='#DA702C')

                xx = np.linspace(-self.vx, self.vx, 100)
                plt.plot(xx, self.forward(self.params, xx), color='#4385BE')

                title = list()
                if 'Loss' in self.title:
                    loss = self.loss_fn(
                        self.params, self.forward,
                        self.dataset.xsmp, self.dataset.ysmp
                        )

                    loss_str = f'Loss: {loss:.2f}'
                    title.append(loss_str)
                if 'Grad' in self.title:
                    grad = self.grad_fn(
                        self.params, self.forward,
                        self.dataset.xsmp, self.dataset.ysmp
                        )

                    grad_str = 'Grad: ' + ' '.join([
                        f'{k}={-v:.2f}'
                        for k, v
                        in grad.items()
                        ])
                    title.append(grad_str)
                plt.title('\n'.join(title))

            elif self.mode == 'Ground truth':
                plt.scatter(self.dataset.x, self.dataset.y,
                            color='lightgrey', alpha=0.5)

                xx = np.linspace(-self.vx, self.vx, 100)
                plt.plot(xx, self.forward(self.dataset.gt, xx), color='#DA702C')
                plt.plot(xx, self.forward(self.params, xx), color='#4385BE')
            else:
                raise ValueError

            plt.show()

    def register_title_cards(self):
        buttons = list()

        mode = ipywidgets.Dropdown(
            options=['Full dataset', 'Subsampled dataset', 'Ground truth'],
            value='Full dataset',
            description='Plot:',
            disabled=False,
        )

        def switch_mode(change):
            self.mode = change.new
            self.display()

        mode.observe(switch_mode, 'value')
        buttons.append(mode)

        title = ipywidgets.Dropdown(
            options=['', 'Loss', 'Grad', 'Loss & Grad'],
            value='',
            description='Title:',
            disabled=False,
        )

        def switch_title(change):
            self.title = change.new
            self.display()

        title.observe(switch_title, 'value')
        buttons.append(title)
        return buttons

    def register_batch_size_buttons(self):
        buttons = list()

        def click_resample(b):
            self.dataset.sample()
            self.display()

        resample = ipywidgets.Button(description="Resample")
        resample.on_click(click_resample)
        buttons.append(resample)

        def change_batch_size_slider(change):
            self.dataset.batch_size = change['new']

        slider = ipywidgets.IntSlider(
            value=self.dataset.batch_size,
            min=1, max=min(self.dataset.n, 50),
            description='Batch size'
        )
        slider.observe(change_batch_size_slider, names='value')

        buttons.append(slider)
        return buttons

    def register_parameter_freezer(self):

        freeze = ipywidgets.RadioButtons(options=['Update', 'Freeze'])

        def observe_freeze(change):
            self.update_params_on_change = change['new'] == 'Update'
            if self.update_params_on_change:
                self.display()

        freeze.observe(observe_freeze, 'value')
        return freeze

    def check_update_params_on_change(self):
        return self.update_params_on_change

    def register_parameters(self):

        def construct_change_fxn(name):

            def on_slider_change(change):
                self.params[name] = change['new']
                if self.check_update_params_on_change():
                    self.display()

            return on_slider_change

        for name, param in self.params.items():
            if param is None:
                continue

            slider = ipywidgets.FloatSlider(
                value=0,
                min=-2, max=2, step=0.25,
                description=name
            )

            slider.observe(construct_change_fxn(name), names='value')
            yield slider


class Dataset:
    def __init__(self, x, y, true_params):
        self.gt = true_params

        assert len(x) == len(y)
        self.n = len(x)
        self.x = x
        self.y = y
        self.batch_size = 5
        self.sample()

    def sample(self):
        assert self.x.size == self.y.size

        idx = np.random.choice(self.x.size, self.batch_size, replace=False)
        self.xsmp = self.x[idx].copy()
        self.ysmp = self.y[idx].copy()

        return


def make_dataset(params, forward, n=100, eps=0.75):
    xx = jnp.array(4 * np.random.rand(n) - 2)
    yy = forward(params, xx) + eps * jnp.array(np.random.randn(len(xx)))
    dataset = Dataset(xx, yy, dict(params.items()))
    return dataset
