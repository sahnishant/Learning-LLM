# Lambda serverless

Learn to set up your own lambda. 

# CNN

## Understanding Data Type

### Images
- Image - Multidimension array (Tensor)
- Each layer (channel) is a 2 dimensional - array (tensor). 
- Grayscale images have only one channel. 
- Advanced images like MRI/ CT Scan/ Medical images/ could have multiple 100+ channels. 

### Text
- Plain text/ CSV/ JSON
- ASCII (128 characters)/ UTF or UTF-8(millions of characters)
- Tokens
    - Tokenizers split the data
    - Computers understand numbers. So we convert words to numbers.
    - 
### Audio
- 1d/ 2D
- Amplitude with time.
    - We can apply fourier transformations to break into individual frequencies. 
- 2D data - STFT like a specrogram time - frequency 
- sampling rate (no of samples per second) - Its the resolution equivalent of images

### 3D data 
- Geometric constructs
- Set of points in space
- Meshes - vertices edges and faces

## Creating Data Set in PyTorch

Dataset - Blueprint to access your data. 
THe class has to have three main methods
- `__init__`: Set up data - load file path or list of labels or audio
- `__Len__`: no of items in the dataset
- `__getitem__`: gives index, loads and prepares each data item. 

    ```
    from torch.utils.data import Dataset

    class CustomDataset(Dataset):
        def __init__(self):
            # Initialization code
            pass

        def __len__(self):
            # Returns the total number of samples
            return 0

        def __getitem__(self, idx):
            # Generates one sample of data
            return None
    ```

## Data Preprocessing
Raw data is messy.
Needs cleaning 
Needs Standardization

- Images
    - Resize to same size
    - Normalize (Pixel values between -1 and 1) 
        - Why? Becasue faster training
    - ToTensor() a pytorch Function 
        - Converts image to tensor
        - Scales pixel values from 0-255 to 0-1
        - RGB in order 
    ```
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]) 
    ```
- Text
    - Tokenize 
    - Make same size    
        - Padding - with blank tokens
        - Truncating - Very long sequence to same size 
    - Embedding - Token to a numberical vector
    ```
    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch):
        texts, labels = zip(*batch)
        texts = [torch.tensor(text) for text in texts]
        texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
        labels = torch.tensor(labels)
        return texts_padded, labels
    ```
- Audio 
    - Resample sampling rate of audio to std value
    - Feature Extraction
    ```transformation = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)```
- 3D Data
    - Normalize
    - Center to Origin
    ```
    def normalize(vertices):
        centroid = vertices.mean(dim=0)
        vertices = vertices - centroid
        max_dist = torch.max(torch.norm(vertices, dim=1))
        vertices = vertices / max_dist
        return vertices
    ```
## Data Augmentation
- creating new training data from existing data 
- making minor changes 
- Goal is to make the model more robust 
- Reduce likelihood to overfit (memorize) the training data.
### Images
- Random Horizontal Flip, Random Rotation, and Color Jitter.
- It helps us get more data
- Add black patch on the face
- remove half the images 
### Text
- new sentences while preserving the original meaning.
- Synonym Replacement
Replaces words with their synonyms. This technique helps the model recognize that different words can have the same meaning. The provided code snippet uses the wordnet corpus to find and replace synonyms.
- Random Insertion
Inserts a random word from the vocabulary into the sentence. This helps the model to become more robust to variations in sentence structure.
### Audio
Audio augmentation modifies the audio signal to expose the model to different acoustic conditions.

- Time Stretching:  
    - Changes the speed of the audio
    - helps the model learn to recognize speech at various speeds.

- Pitch Shifting: 
    - Changes the pitch without changing its speed. 
    - This helps the model become more robust to variations in speaking pitch.
- Adding Background Noise
    - Adds random noise to the audio. 
    - Crucial step for models
    - in real-world environments: background noise.

### 3D Data
- 3D data augmentation helps models recognize objects from different angles and under different conditions.
- Random Rotation
- Scaling: Changes the size of the 3D object.
-  Adding Noise Adds random noise to the vertices - This helps the model learn to recognize the object even when some of its points are slightly misplaced.


## Building First Neural Networks
