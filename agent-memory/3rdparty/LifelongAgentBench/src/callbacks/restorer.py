from .callback import Callback


class CallbackRestorer:
    @staticmethod
    def restore(callback_dict: dict[str, Callback]) -> None:
        for callback_id, callback in callback_dict.items():
            callback.restore_state()
