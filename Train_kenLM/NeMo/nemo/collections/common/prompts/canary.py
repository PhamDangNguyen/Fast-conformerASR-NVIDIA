from typing import Any

import torch
from lhotse import CutSet, MonoCut
from lhotse.cut import MixedCut
from lhotse.utils import ifnone

from nemo.collections.common.prompts.fn import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.common.tokenizers.canary_tokenizer import (
    CANARY_BOS,
    CANARY_EOS,
    CANARY_NOPNC,
    CANARY_PNC,
    CANARY_SPECIAL_TOKENIZER,
)


class CanaryPromptFormatter(PromptFormatter):
    NAME = "canary"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"{CANARY_BOS}|source_lang||task||target_lang||pnc|",
            "slots": {
                "source_lang": Modality.Text,
                "task": Modality.TextLiteral(
                    "asr", "ast", "translate", "transcribe", "s2t_translation", "<|transcribe|>", "<|translate|>"
                ),
                "target_lang": Modality.Text,
                "pnc": Modality.TextLiteral(
                    "yes", "no", "true", "True", "false", "False", "1", "0", "pnc", "nopnc", "<|pnc|>", "<|nopnc|>"
                ),
            },
        },
        OUTPUT_ROLE: {
            "template": f"|text|{CANARY_EOS}",
            "slots": {
                "text": Modality.Text,
            },
        },
    }

    def _validate_slot_values(self, expected: dict[str, Modality], received: dict[str, Any]) -> None:
        if "taskname" in received and "task" not in received:
            received["task"] = received.pop("taskname")
        return super()._validate_slot_values(expected=expected, received=received)

    def encode_turn(self, prompt_template: str, expected_slots: dict, slot_values: dict) -> list[int]:
        # This method handles a level of indirection for Canary.
        # It maps values provided in trcfg to the actual special tokens
        # expected to be present in canary prompt.
        # It used to be done in prompt_format_fn inside Dataset class corresponding to Canary,
        # but we are not using it here anymore.
        # This maps things such as '|task|: "asr"' to '|TASK|: "<|transcribe|>"'.
        slot_values = map_manifest_values_to_special_tokens(slot_values)
        return super().encode_turn(
            prompt_template=prompt_template, expected_slots=expected_slots, slot_values=slot_values
        )


def map_manifest_values_to_special_tokens(slot_values: dict[str, str]) -> dict[str, str]:
    slot_values = slot_values.copy()

    any_special_token_present = False

    for k in ("source_lang", "target_lang"):
        if k in slot_values and not ((v := slot_values[k]).startswith("<|") and v.endswith("|>")):
            slot_values[k] = "<|" + slot_values[k] + "|>"
            any_special_token_present = True

    k = "pnc"
    if k in slot_values and slot_values[k] not in (CANARY_PNC, CANARY_NOPNC):
        slot_values[k] = CANARY_PNC if slot_values[k] in ("yes", "1", "True", "true", "pnc") else CANARY_NOPNC
        any_special_token_present = True

    # Note: we re-map 'taskname' to 'task' for compatibility with earlier versions of Canary training.
    for k in ("task", "taskname"):
        if k in slot_values and slot_values[k] not in ("<|transcribe|>", "<|translate|>"):
            if slot_values[k] in {"translate", "ast", "s2t_translation"}:
                slot_values["task"] = "<|translate|>"
            elif slot_values[k] in {"transcribe", "asr"}:
                slot_values["task"] = "<|transcribe|>"
            else:
                assert False, f"Task {slot_values[k]} invalid task for slot {k}"
            any_special_token_present = True

    # Auto-inject which tokenizer to look up in CanaryTokenizer if not provided,
    # and slots for this turn correspond to user prompt.
    if any_special_token_present and PromptFormatter.PROMPT_LANGUAGE_SLOT not in slot_values:
        slot_values[PromptFormatter.PROMPT_LANGUAGE_SLOT] = CANARY_SPECIAL_TOKENIZER

    return slot_values


@registered_prompt_format_fn
def canary(
    cuts: CutSet, tokenizer: TokenizerSpec
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Prepend and append control tokens to the token sequence as per Canary format.

    We use the following special tokens:
    * <|startoftranscript|>
    * <|transcribe|>
    * <|translate|>
    * <|nopnc|>
    * <|pnc|>
    * <|endoftext|>
    * <|LANG|> - for each supported language.
    * <|nospeech|>

    The prompt format syntax is as follows:

        <|startoftranscript|> [ <|nospeech|> | <|LANG|> [ <|transcribe|> | <|translate|> ] <|LANG|> [ <|pnc|> | <|nopnc|> ] TEXT <|endoftext|> ]

    Where expression ``[ a | b ]`` denotes expression ``a`` or expression ``b``, and can be nested.
    Note that ``<|LANG|>`` appears twice: the first occurrence is for the "source" language
    (i.e., spoken language in the recording) and the second occurrence is for the "target" language
    (i.e., the language in which we are going to output the text).
    """
    formatter = CanaryPromptFormatter(tokenizer)

    prompts_with_answers, prompts, answers = [], [], []
    for cut in cuts:
        if isinstance(cut, MixedCut):
            cut = cut._first_non_padding_cut
        if not isinstance(cut, MonoCut):
            raise TypeError(
                f"Expected input audio to have a single channel (required MonoCut/MixedCut, but we received: {cut=})"
            )

        # first, validate the utterance
        expected_slots = set(formatter.get_slots("user"))
        missing_keys = expected_slots - set(cut.custom)
        if "task" in missing_keys and "taskname" in cut.custom:
            # Compatibility with "old" Canary manifest format.
            # For compatbility with inference options, this slot is now called "task".
            cut.custom["task"] = cut.custom["taskname"]
            missing_keys.remove("task")
        if missing_keys:
            raise RuntimeError(
                f"We found cut with ID {cut.id} that is missing the following keys: {missing_keys}"
                f"Please ensure that every utterance in the input manifests contains these keys."
            )

        turns = [
            dict(
                role="user",
                slots={
                    **{slot: cut.custom[slot] for slot in expected_slots},
                    formatter.PROMPT_LANGUAGE_SLOT: CANARY_SPECIAL_TOKENIZER,
                },
            )
        ]
        # If data has no transcript, create empty response with <eos> only.
        text = ' '.join(s.text for s in cut.supervisions if s.text is not None)
        turns.append(
            dict(
                role="assistant",
                slots={
                    "text": text,
                    formatter.PROMPT_LANGUAGE_SLOT: ifnone(
                        cut.supervisions[0].language, cut.custom.get("target_lang")
                    ),
                },
            ),
        )
        encoded = formatter.encode_dialog(turns)
        prompts_with_answers.append(encoded["input_ids"])
        prompts.append(encoded["context_ids"])
        if "answer_ids" in encoded:
            assert (
                encoded["answer_ids"][-1].item() == formatter.tokenizer.eos
            ), f"Expected the last token in answer_ids to be EOS, but we got {encoded['answer_ids']=}"
            answers.append(encoded["answer_ids"][:-1])  # Strip Canary's EOS
        else:
            answers.append([])

    return prompts_with_answers, prompts, answers
