
class Callback(object):
    """A base Callback class."""

    def __init__(self):
        self.model = None
        self.params = {}

    def set_model(self, model):
        self.model = model

    def set_params(self, params):
        self.params = params

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_iteration_begin(self, iteration):
        pass

    def on_iteration_end(self, iteration, logged_vars=None):
        pass


class History(Callback):
    """A simple History callback. Stores the history of the whole training."""

    def on_train_begin(self):
        self.history = {}
        self.iterations = []

    def on_iteration_end(self, iteration, logged_vars=None):
        self.iterations.append(iteration)
        for var, val in logged_vars.items():
            self.history.setdefault(var, []).append(val)


class ParamsTerminator(Callback):
    def on_train_begin(self):
        self.min_improvement = self.params['stop_threshold']
        self.min_iterations = self.params['min_iterations']
        self.max_iterations = self.params['max_iterations']

    def on_iteration_end(self, iteration, logged_vars=None):
        improvement = logged_vars.get('improvement') or 0

        self.model.stop_training = True
        if improvement > self.min_improvement or \
                iteration < self.min_iterations + 1:
            self.model.stop_training = False

        if iteration >= self.max_iterations:
            self.model.stop_training = True


class CallbackList(object):
    """A container for a list of callbacks."""

    def __init__(self, callbacks=None, params=None, model=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.params = params
        self.set_params(params)

        self.set_model(model)
        self.model = model

    def append(self, callback):
        self.callbacks.append(callback)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_iteration_begin(self, iteration):
        for callback in self.callbacks:
            callback.on_iteration_begin(iteration)

    def on_iteration_end(self, iteration, logged_vars=None):
        logged_vars = logged_vars or {}
        for callback in self.callbacks:
            callback.on_iteration_end(iteration, logged_vars)
