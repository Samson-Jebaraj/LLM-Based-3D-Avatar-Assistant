use log::{info, LevelFilter};
use tch::*;
use std::collections::HashMap;

/*

Copyright neuralharbour.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This module contains code adapted from `transformers`.
Copyright and license details can be found in `NOTICE.md`.

*/

pub struct CollatorMASSIVEIntentClassSlotFill {
    /*
    Data collator for the MASSIVE intent classification and slot tasks
    Based on: https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/data/data_collator.py#L212

    :param tokenizer: The tokenizer
    :type tokenizer: transformers.PreTrainedTokenizerFast
    :param padding: True or 'longest' pads to longest seq in batch, 'max_length' to the specified
                    max_length, and False or 'do_not_pad' to not pad (default)
    :type padding: bool, str, or transformers.file_utils.PaddingStrategy
    :param max_length: max length for truncation and/or padding (optional)
    :type max_length: int
    :param pad_to_multiple_of: set the padding such that sequence is multiple of this (optional)
    :type pad_to_multiple_of: int
    */
    tokenizer: Box<dyn Fn(&[String]) -> Vec<HashMap<String, Tensor>>>,
    max_length: i64,
    padding: String,
    pad_to_multiple_of: Option<i64>,
    col_chk: usize,
}

impl CollatorMASSIVEIntentClassSlotFill {
    pub fn new<F>(
        tokenizer: F,
        max_length: i64,
        padding: String,
        pad_to_multiple_of: Option<i64>,
    ) -> Self
    where
        F: 'static + Fn(&[String]) -> Vec<HashMap<String, Tensor>>,
    {
        Self {
            tokenizer: Box::new(tokenizer),
            max_length,
            padding,
            pad_to_multiple_of,
            col_chk: 0,
        }
    }

    pub fn call(&mut self, batch: Vec<HashMap<String, Tensor>>) -> HashMap<String, Tensor> {
        let utterances: Vec<String> = batch
            .iter()
            .map(|item| item["utt"].to_string())
            .collect();

        let mut tokenized_inputs = (self.tokenizer)(&utterances);

        for (i, entry) in batch.iter().enumerate() {
            let label = entry["slots_num"].iter().map(|v| *v as i64).collect::<Vec<i64>>();
            let word_ids = tokenized_inputs[i]["word_ids"].iter().map(|&x| x as Option<i64>).collect::<Vec<_>>();
            let mut previous_word_idx = None;
            let mut label_ids = Vec::new();

            for word_idx in word_ids {
                match word_idx {
                    None => label_ids.push(-100),
                    Some(idx) if Some(idx) != previous_word_idx => {
                        label_ids.push(label[idx as usize]);
                        previous_word_idx = Some(idx);
                    }
                    _ => label_ids.push(-100),
                }
            }

            if self.col_chk != 0 {
                info!(
                    "Collator Check! utt: {:?}; intent label: {:?}; slot labels: {:?}; tokenized utt: {:?}; label_ids: {:?}",
                    entry["utt"],
                    entry["intent_num"],
                    entry["slots_num"],
                    tokenized_inputs[i],
                    label_ids
                );
                self.col_chk += 1;
            }

            let slots_num = tokenized_inputs[i].entry("slots_num".to_string()).or_insert_with(|| {
                Tensor::of_slice(&label_ids).to(Device::Cpu)
            });
            *slots_num = Tensor::of_slice(&label_ids).to(Device::Cpu);
        }

        let sequence_length = tokenized_inputs[0]["input_ids"].size()[1];

        for input in &mut tokenized_inputs {
            let slots_num = input.get_mut("slots_num").unwrap();
            let label_len = slots_num.size()[0];
            if self.padding == "right" {
                *slots_num = Tensor::cat(
                    &[slots_num.copy(), Tensor::of_slice(&vec![-100; (sequence_length - label_len) as usize])],
                    0,
                );
            } else {
                *slots_num = Tensor::cat(
                    &[Tensor::of_slice(&vec![-100; (sequence_length - label_len) as usize]), slots_num.copy()],
                    0,
                );
            }
        }

        let mut result = HashMap::new();
        result.insert(
            "input_ids".to_string(),
            Tensor::stack(
                &tokenized_inputs
                    .iter()
                    .map(|input| input["input_ids"].copy())
                    .collect::<Vec<_>>(),
                0,
            ),
        );
        result.insert(
            "intent_num".to_string(),
            Tensor::of_slice(
                &batch
                    .iter()
                    .map(|item| item["intent_num"].int64_value(&[]))
                    .collect::<Vec<_>>(),
            ),
        );

        result
    }
}
