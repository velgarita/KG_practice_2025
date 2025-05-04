import argparse
import logging

from models.classic_model import train_classic
from models.llm_model import train_llm
from utils import load_dataset

logging.basicConfig(
    filename='results.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def main():
    parser = argparse.ArgumentParser(description="Train relation classification model.")
    parser.add_argument("--model", choices=["classic", "llm"], required=True, help="Model type to use.")
    args = parser.parse_args()

    logging.info("Loading dataset...")
    df = load_dataset()

    logging.info(f"Training {args.model} model...")
    if args.model == "classic":
        train_classic(df)
    elif args.model == "llm":
        train_llm(df)

if __name__ == "__main__":
    main()
    