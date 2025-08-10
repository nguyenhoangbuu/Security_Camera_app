import telegram

my_token = "XXX"

# Tạo bot
bot = telegram.Bot(token=my_token)


# Gửi thử text message
bot.sendPhoto(chat_id="XXX", photo = open("b.jpeg","rb"), caption = "Có súng, nguy hiêm!")
