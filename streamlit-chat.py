from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, BitsAndBytesConfig, TextStreamer
import streamlit as st

# ---------- PLEASE READ CAREFULLY ----------
# I had to Quantize Blenderbot 3B; My GPU with 6GB VRAM was not happy loading the model without memory optimization.


hf_card = 'facebook/blenderbot-3B'

st.set_page_config(
    page_title= 'WellnessSquad Chat',
    page_icon='🤖'
)

with st.spinner('Processing..'):
    quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    low_cpu_mem_usage = True
    )

    model = BlenderbotForConditionalGeneration.from_pretrained(
        hf_card,
        quantization_config = quantization_config
    )

    tokenizer = BlenderbotTokenizer.from_pretrained(hf_card, padding_side='left')
    # Not Always Necessary; Most LLMs don't have a pad token by default
    tokenizer.pad_token = tokenizer.eos_token 
    # Setting eos_token_id in model.config tells it when to stop generation a sentence.
    model.config.eos_token_id = tokenizer.eos_token_id


message_template = [
    {
        "role": "system",
        "content": "You are a kind and empathetic person. You listen and offer advice when applicable. You do not judge people and lend people an open ear to vent.",
        },
    {
        "role": "user", 
        "content": None
    },
]


if "messages" not in st.session_state:
    st.session_state.messages =  [
    {
        "role": "system",
        "content": "Hello 👋, welcome to WellnessSquad chat! You are connected with Cyrus 😁",
        "name": "ai"
    }
]
    

for message in st.session_state.messages:
    with st.chat_message(message['name']):
        st.markdown(message['content'])

if prompt := st.chat_input('Enter your message here'):
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append(
        {
            'role' : 'user',
            'content' : prompt,
            "name": "user"
        }
    )
    message_template[1]['content'] = prompt
    
    user_input = tokenizer.apply_chat_template(
        message_template,
        tokenize=True,
        add_generation_prompt = True,
        return_tensors = 'pt'
    ).cuda()


    generated_ids = model.generate(
        user_input,
        max_new_tokens = float('inf'),
        temperature = 0.7,
        repetition_penalty = 2.0
    )

    response = tokenizer.batch_decode(
        generated_ids,
        clean_up_tokenization_spaces=True, 
        skip_special_tokens=True
    )[0]

    with st.chat_message('ai'):
        st.markdown(response)
    st.session_state.messages.append(
        {
            'role' : 'system',
            'content': response,
            "name": "ai"
        }
    )
