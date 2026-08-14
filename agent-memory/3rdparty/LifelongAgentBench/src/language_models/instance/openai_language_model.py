from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import openai
import os
from typing import Any, Optional, Sequence, Mapping, TypeGuard

from src.language_models.language_model import LanguageModel
from src.typings import (
    Role,
    ChatHistoryItem,
    LanguageModelContextLimitException,
    ChatHistory,
)
from src.utils import RetryHandler, ExponentialBackoffStrategy


class OpenaiLanguageModel(LanguageModel):
    """
    To keep the name of the class consistent with the name of file, use OpenaiAgent instead of OpenAIAgent.
    """

    def __init__(
        self,
        model_name: str,
        role_dict: Mapping[str, str],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        maximum_prompt_token_count: Optional[int] = None,
    ):
        """
        max_prompt_tokens: The maximum number of tokens that can be used in the prompt. It can be used to set the
            context limit manually. If it is set to None, the context limit will be the same as the context length of
            the model selected.
        """
        super().__init__(role_dict)
        self.model_name = model_name
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.maximum_prompt_token_count = maximum_prompt_token_count

    @staticmethod
    def _is_valid_message_list(
        message_list: list[Mapping[str, str]],
    ) -> TypeGuard[list[ChatCompletionMessageParam]]:
        for message_dict in message_list:
            if (
                "role" not in message_dict.keys()
                or "content" not in message_dict.keys()
            ):
                return False
        return True

    @RetryHandler.handle(
        max_retries=3,
        retry_on=(openai.BadRequestError,),
        waiting_strategy=ExponentialBackoffStrategy(interval=(None, 60), multiplier=2),
    )
    def _get_completion_content(
        self,
        message_list: Sequence[ChatCompletionMessageParam],
        inference_config_dict: Mapping[str, Any],
    ) -> Sequence[str]:
        """
        I do not know what will happen when the context limit is reached. According to OpenAI documents, there is no
        type of error for the context limit. So I guess the model will return an empty response when the context limit
        is reached. This may be a potential bug and I apologize in advance.
        There are also some issues on GitHub state that the model will raise openai.BadRequestError in this situation.
        So I also handle this error in the code.
        Reference:
        https://platform.openai.com/docs/guides/error-codes#python-library-error-types
        https://github.com/run-llama/llama_index/discussions/11889
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_list,
                **inference_config_dict,
            )
        except openai.BadRequestError as e:
            if "context length" in str(e):
                # Raise LanguageModelContextLimitException to skip retrying.
                raise LanguageModelContextLimitException(
                    f"Model {self.model_name} reaches the context limit. "
                )
            else:
                # Raise the original exception to retry.
                raise e
        if (
            completion.usage is not None
            and self.maximum_prompt_token_count is not None
            and completion.usage.prompt_tokens > self.maximum_prompt_token_count
        ):
            raise LanguageModelContextLimitException(
                f"Model {self.model_name} reaches the context limit. "
                f"Current prompt tokens: {completion.usage.prompt_tokens}. "
                f"Max prompt tokens: {self.maximum_prompt_token_count}."
            )
        content_list: list[str] = []
        content_all_invalid_flag: bool = True
        for choice in completion.choices:
            content = choice.message.content
            if content is not None and len(content) > 0:
                content_all_invalid_flag = False
            content_list.append(content or "")
        if content_all_invalid_flag:
            raise LanguageModelContextLimitException(
                f"Model {self.model_name} returns empty response. The context limit may be reached."
            )
        return content_list

    def _inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Mapping[str, Any],
        system_prompt: str,
    ) -> Sequence[ChatHistoryItem]:
        """
        system_prompt: It is usually called as system_prompt. But in OpenAI documents, it is called as developer_prompt.
            But in practice, using `message_list = [{"role": "developer", "content": self.system_prompt}]` will raise an
            error. So all after all, I call it as system_prompt.
            Reference:
            https://platform.openai.com/docs/guides/text-generation#messages-and-roles
            https://platform.openai.com/docs/api-reference/chat/create
        inference_config_dict: Other config for OpenAI().chat.completions.create.
            e.g.:
            max_completion_tokens: The maximum number of tokens that can be generated in the chat completion. Notice
                that max_tokens is deprecated.
            Reference:
            https://platform.openai.com/docs/api-reference/chat/create#chat-create-max_completion_tokens
        """
        # region Construct batch_message_list
        message_list_prefix: list[ChatCompletionMessageParam]
        if len(system_prompt) > 0:
            message_list_prefix = [{"role": "system", "content": system_prompt}]
        else:
            message_list_prefix = []
        batch_message_list: list[Sequence[ChatCompletionMessageParam]] = []
        for chat_history in batch_chat_history:
            conversion_result = self._convert_chat_history_to_message_list(chat_history)
            assert OpenaiLanguageModel._is_valid_message_list(conversion_result)
            batch_message_list.append(message_list_prefix + conversion_result)
        # endregion
        # region Generate output
        output_str_list: list[str] = []
        for message_list in batch_message_list:
            output_str_list.extend(
                self._get_completion_content(message_list, inference_config_dict)
            )
        # endregion
        # region Convert output to ChatHistoryItem
        return [
            ChatHistoryItem(role=Role.AGENT, content=output_str)
            for output_str in output_str_list
        ]
        # endregion
