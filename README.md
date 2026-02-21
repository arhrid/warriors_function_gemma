# WarriorsPersonalAssistant
Personal voice agent: Handles WhatsApp calls/messages and sends email. Data is stored locally, using FunctionGemma to make tool calls.

The purpose of this document is to capture our usecase and the process. 

## Usecase Description
1. there is an active voice listening in our mac. 
2. The user would say "hey warrior, message xxx.xxx.xxxx number on my whatsapp that we are set for dinner at XYZ for 5pm today. Then, email his reponse to xxx@gmail.com.

## Our Architecture proposal for Internal logic of the generate_hybrid method in main.py
