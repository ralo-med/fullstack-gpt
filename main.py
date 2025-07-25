import tiktoken
from dotenv import load_dotenv
load_dotenv()


def main():
    print(tiktoken.encoding_for_model("gpt-4o"))


if __name__ == "__main__":
    main()
