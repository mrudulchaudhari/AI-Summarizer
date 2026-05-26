from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# Load model and tokenizer
model_name = "facebook/bart-large-cnn"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully!")


def summarize_article(article, max_length, min_length):
    if len(article.strip()) == 0:
        return "Please enter an article."

    # Tokenize input article
    inputs = tokenizer(
        article,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode summary
    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


# Gradio Interface
interface = gr.Interface(
    fn=summarize_article,
    inputs=[
        gr.Textbox(
            lines=15,
            placeholder="Paste your article here...",
            label="Article"
        ),
        gr.Slider(
            minimum=75,
            maximum=300,
            value=120,
            step=5,
            label="Max Summary Length"
        ),
        gr.Slider(
            minimum=10,
            maximum=75,
            value=30,
            step=5,
            label="Min Summary Length"
        )
    ],
    outputs=gr.Textbox(
        lines=8,
        label="Generated Summary"
    ),
    title="AI-Powered Article Summarizer",
    description="Summarize long articles using Hugging Face Transformers and Gradio"
)


# Launch app
if __name__ == "__main__":
    interface.launch()