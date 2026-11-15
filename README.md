# Text Mining - Assignment 2: Sequence Labelling (Archaeo-NER)

This project fine-tunes a BERT Transformer for Named Entity Recognition (NER) on archaeological texts using the 'fold 1' split of [Archaeo-NER-data-English](https://github.com/alexbrandsen/Archaeo-NER-data-English).

The model predicts six archaeological entity types using strict IOB2 tagging
- `ART` = Artefact
- `CON` = Context / Feature
- `LOC` = Location
- `MAT` = Material
- `PER` = Person
- `SPE` = Species

## Structure

The codebase is split into modular components:

1. **[`data_loader.py`](data_loader.py)**:
   Reads 3-column IOB text rows (`word`, `pos`, `tag`), groups sentences by blank lines, and maps tags to integer IDs and converts the splits into a Hugging Face `DatasetDict`.

2. **[`tokenization.py`](tokenization.py)**:
   Implements the Hugging Face subword alignment. The first subword of each word gets the word's NER label, while subsequent subwords and special tokens receive `-100`.

3. **[`evaluation.py`](evaluation.py)**:
   Computes metrics for all `B-` and `I-` token labels, exact entity matches per entity type and overall Entity Micro F1 and Macro F1 scores.

4. **[`trainer.py`](trainer.py)**:
   Configures `AutoModelForTokenClassification` and `TrainingArguments`, trains using `Trainer` and runs test set predictions.

5. **[`run_ner_pipeline.py`](run_ner_pipeline.py)**:
   Clean top-level coordinator parsing arguments and calling each module in sequence.

---

## Usage

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download data using**
   ```bash
   python download_data.py
   ```

4. **Run the full training & evaluation pipeline**
   ```bash
   python run_ner_pipeline.py
   ```

5. **You can also evaluate an existing model checkpoint without re-training**
   ```bash
   python run_ner_pipeline.py --eval_only --model_path outputs/baseline/final_model
   ```
