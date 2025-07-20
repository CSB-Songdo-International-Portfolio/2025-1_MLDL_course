# pip install transformers torch gradio
# 에러나면 accelerate 설치

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 모델 이름
model_name = "Qwen/Qwen3-0.6B"

# 모델 및 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 채팅 히스토리 저장
chat_history = []

# 추론 함수
def predict(user_input, history_display):
    global chat_history

    # 사용자 메시지 추가
    chat_history.append({"role": "user", "content": user_input})

    # 채팅 템플릿 적용
    prompt = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True
    )

    # 토크나이즈
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 생성 설정
    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        max_new_tokens=512
    )

    # 응답 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )

    # 출력 디코딩
    output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    output_text = output_text.strip()

    # 어시스턴트 응답 저장
    chat_history.append({"role": "assistant", "content": output_text})

    # Gradio에 표시할 히스토리 업데이트
    history_display.append((user_input, output_text))
    return "", history_display

# Gradio 인터페이스 구성
with gr.Blocks(title="Qwen 0.6B Chatbot") as demo:
    gr.Markdown("## 🤖 Qwen 0.6B Chatbot (powered by Gradio)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="입력", placeholder="질문을 입력하세요...", lines=1)
    clear = gr.Button("초기화")

    # 입력 제출 이벤트
    msg.submit(predict, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], []), None, [chatbot])

# 앱 실행
if __name__ == "__main__":
    demo.launch()
