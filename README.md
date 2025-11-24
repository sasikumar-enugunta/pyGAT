# pyGAT - Graph Attention Network for Named Entity Recognition

This project implements a Graph Attention Network (GAT) for Named Entity Recognition (NER) on invoice documents. The model combines LSTM embeddings with graph attention layers to perform sequence labeling.

## Project Structure

```
pyGAT/
├── train.py              # Main training script
├── model.py              # GAT model definition
├── layers.py             # Graph attention layer implementation
├── utils.py              # Data loading and utility functions
├── utils2.py            # Additional utilities (OCR, color detection, etc.)
└── data/
    └── invoice/          # Data files directory
```

## Requirements

### Python Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install torch torchvision numpy pandas scikit-learn opencv-python pillow easyocr webcolors extcolors
```

### Data Files

The project requires the following data files in `data/invoice/`:

1. **`invoice.final_feature_embeddings_updated_1`** - Feature embeddings file (already present)
2. **`invoice.final_coordinates_1`** - Coordinates file (already present)
3. **`invoice.final_edge_embeddings_updated`** - Edge embeddings file (missing - needs to be created)
4. **`GoogleNews-vectors-negative300.bin`** - Pre-trained Word2Vec embeddings (missing - needs to be downloaded)

#### Downloading Word2Vec Embeddings

Download the Google News Word2Vec embeddings (1.6GB):

```bash
cd data/invoice/
wget https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# Or use direct link if available
# The file should be named: GoogleNews-vectors-negative300.bin
```

**Note:** The Google News Word2Vec file is large (~1.6GB). You may need to download it from Google's official source or use an alternative pre-trained embedding file.

## Running the Project

### Basic Usage

Run the training script with default parameters:

```bash
python train.py
```

### Command Line Arguments

The script supports various command-line arguments:

```bash
python train.py [OPTIONS]
```

**Training Options:**
- `--no-cuda`: Disable CUDA (use CPU only)
- `--epochs N`: Number of training epochs (default: 10000)
- `--lr FLOAT`: Learning rate (default: 0.001)
- `--weight_decay FLOAT`: Weight decay for L2 regularization (default: 5e-4)
- `--dropout FLOAT`: Dropout rate (default: 0.33)
- `--alpha FLOAT`: Alpha for LeakyReLU (default: 0.2)
- `--seed INT`: Random seed (default: 72)

**Model Architecture Options:**
- `--embedding_dim INT`: LSTM embedding dimension (default: 500)
- `--hidden1 INT`: First hidden layer size (default: 256)
- `--hidden2 INT`: Second hidden layer size (default: 128)
- `--hidden3 INT`: Third hidden layer size (default: 64)
- `--hidden4 INT`: Fourth hidden layer size (default: 32)

### Example Commands

Train with custom parameters:

```bash
python train.py --epochs 100 --lr 0.0005 --dropout 0.4
```

Train on CPU only:

```bash
python train.py --no-cuda
```

Train with custom architecture:

```bash
python train.py --hidden1 512 --hidden2 256 --hidden3 128 --hidden4 64
```

## How It Works

1. **Data Loading**: The script loads invoice data from feature embeddings, edge embeddings, and coordinate files.

2. **Data Splitting**: Data is split into training (80%) and test (20%) sets.

3. **Model Training**: 
   - The GAT model processes each document
   - Uses LSTM for sequence encoding
   - Applies graph attention layers for spatial relationships
   - Uses Conditional Random Fields (CRF) for sequence labeling

4. **Evaluation**: After training, the model computes precision, recall, and F1-score on the test set.

## Model Architecture

The model consists of:
- **Embedding Layer**: Word embeddings using pre-trained Word2Vec
- **LSTM Layer**: Bidirectional LSTM for sequence encoding
- **Graph Attention Layers**: Two GAT layers for spatial relationship modeling
- **CRF Layer**: Conditional Random Fields for sequence labeling

## Troubleshooting

### Missing Data Files

If you encounter file not found errors:
- Ensure all required data files are in `data/invoice/`
- Check that file names match exactly (case-sensitive)
- Verify the Word2Vec file is downloaded and in the correct location

### Memory Issues

If you run out of memory:
- Reduce batch size (currently processes one document at a time)
- Use `--no-cuda` to run on CPU (slower but uses less GPU memory)
- Reduce hidden layer dimensions

### Import Errors

If you get import errors:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (Python 3.6+ recommended)
- Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`

## Notes

- The current implementation trains for 10 epochs (hardcoded in `train.py` line 108)
- Training uses Adam optimizer
- The model processes documents sequentially (one at a time)
- Evaluation metrics include confusion matrix, precision, recall, and F1-score

