# CharlesHendrix

CharlesHendrix is a project that uses a genetic algorithm to generate musical compositions. The fitness of each composition is evaluated by a pre-trained machine learning model, guiding the evolutionary process toward creating aesthetically pleasing music.

## Project Overview

This project combines concepts from evolutionary computation and machine learning to explore algorithmic music generation. The core of the project is a genetic algorithm that evolves populations of musical "genomes." A machine learning model, trained on user feedback, acts as the fitness function, predicting a score for each generated genome.

## Directory Structure

The project is organized into the following directories:

*   `data/`: Contains raw and processed data, including user feedback, generated compositions, and feature notes.
*   `documentation/`: Holds project documentation.
*   `model/`: Stores the serialized machine learning model, scaler, and other necessary objects.
*   `src/`: Contains all the Python source code.
    *   `data_processing/`: Scripts for data extraction and feature engineering.
    *   `explainability/`: Tools for interpreting the machine learning model's predictions.
    *   `genetic_algorithm/`: The core implementation of the genetic algorithm, including fitness evaluation and genome handling.
    *   `test_models/`: Scripts for evaluating and comparing different machine learning models.

## Getting Started

### Prerequisites

*   Python 3.10 or higher
*   pip

### Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/SugarStoneMaster/CharlesHendrix.git
    cd CharlesHendrix
    ```

2.  Install the required Python packages using `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To run the genetic algorithm and generate a new musical composition, execute the main script:

```bash
python src/main.py
```

The script will run the evolutionary process and save the resulting composition.

## How It Works

1.  **Genetic Algorithm**: The process starts with an initial population of random musical genomes.
2.  **Fitness Evaluation**: Each genome is converted into a set of musical features. These features are fed into a pre-trained machine learning model (`model/model.joblib`) which outputs a fitness score.
3.  **Selection, Crossover, and Mutation**: Genomes with higher fitness scores are more likely to be selected for reproduction. New genomes are created through crossover (combining parts of two parent genomes) and mutation (randomly altering a genome).
4.  **Evolution**: This cycle repeats for a set number of generations, with the population's overall fitness improving over time. The best genome from the final generation is selected as the result.