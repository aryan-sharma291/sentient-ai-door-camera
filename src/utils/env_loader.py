from dotenv import load_dotenv
import os



def load_api_keys():
    load_dotenv()
    return {
        "GOOGLE_APPLICATION_CREDENTIALS": os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        "TELEGRAM_BOT_TOKEN": os.getenv('TELEGRAM_BOT_TOKEN'),
        "TELEGRAM_CHAT_ID": os.getenv('TELEGRAM_CHAT_ID')
    }