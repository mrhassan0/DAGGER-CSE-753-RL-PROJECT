# Stanford OCR dataset

## File

- `letter.data`: full tab-delimited Stanford OCR handwritten-word dataset.
- SHA-256: `535363d0b7bc34246b1f8731719540f526b6807b12dda5d430d011d160bc7882`

The file was copied unchanged from
`../../../../Imitiation-Learning/Dataset/letter.data` so the Final Project is
self-contained.

## Row schema

```text
id, letter, next_id, word_id, position, fold, pixel_0, ..., pixel_127
```

- `letter` is one of `a` through `z`.
- `next_id` is `-1` for the last letter of a word.
- `fold` is from 0 through 9.
- The remaining 128 values are binary pixels from a 16-by-8 letter image.

## Verified contents

- 52,152 letter rows
- 6,877 words
- 26 classes
- 10 folds
- word lengths from 3 to 14
- average word length approximately 7.58

## Experimental split

- Training: folds 0--8
- Testing: fold 9

Always split by complete words. For free-running evaluation, the previous-letter
feature must come from the learner's previous prediction, not the ground-truth
letter.
