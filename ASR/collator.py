import torch
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor_pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract audio input values and text labels
        input_values = [feature["input_values"] for feature in features]
        labels = [feature["labels"] for feature in features]

        # Pad audio waveforms to the max length in this batch
        input_values_padded = torch.nn.utils.rnn.pad_sequence(
            input_values, batch_first=True, padding_value=0.0
        )

        # Pad text labels with the pad/ignore token id
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.processor_pad_token_id
        )

        return {
            "input_values": input_values_padded,
            "labels": labels_padded
        }