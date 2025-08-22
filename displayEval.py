import os
# Set environment variables for Hugging Face cache
os.environ["HF_HOME"] = "/scratch/rs02358/huggingface"
os.environ["TMPDIR"] = "/scratch/rs02358/tmp"

import shutil
import tempfile
import gradio as gr
import pandas as pd
from datetime import datetime
from evalUtils import evaluate_video, get_video_info #, convert_codec


def _unpack_temcon(tem_list):
    """
    tem_list: list[float] expected length 7
      [0] -> UI/Temporal Consistency
      [1:] -> CSV columns in order: 4f, 8f, 12f, 16f, 20f, 24f
    Returns:
      ui_value (float), extras (dict)
    """
    # フォールバック（不足要素は None）
    vals = list(tem_list) if isinstance(tem_list, (list, tuple)) else [tem_list]
    while len(vals) < 7:
        vals.append(None)

    ui_value = vals[0]
    extras = {
        "TC per 4f":  vals[1],
        "TC per 8f":  vals[2],
        "TC per 12f": vals[3],
        "TC per 16f": vals[4],
        "TC per 20f": vals[5],
        "TC per 24f": vals[6],
    }
    return ui_value, extras


def run_evaluation(video_path, prompt, fps):
    # Run your evaluation function
    result = evaluate_video(video_path, prompt, fps=fps)

    tem_ui, _extras = _unpack_temcon(result.get("Tem-Con", [None]))

    return (
        f"{result['clip_score']:.4f}",
        f"{result['pick_score']:.4f}",
        (f"{tem_ui:.4f}" if tem_ui is not None else ""),   # guard None
        f"{result['embedding_distance']:.4f}",
    )


def update_video_info(video_path):
    if not video_path or not os.path.exists(video_path):
        return None, "", "", "", "", ""
    
    # Convert video codec for gradio compatibility
    # video_path = convert_codec(video_path)

    filename, duration, frame_count, actual_fps, resolution = get_video_info(video_path)
    return video_path, filename, duration, frame_count, actual_fps, resolution


def update_result_table(video, prompt, fps, history_df):
    video_path = video
    result = evaluate_video(video_path, prompt, fps=fps)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Get video information
    filename, duration, frame_count, actual_fps, resolution = get_video_info(video_path)

    # --- CHANGED: unpack Tem-Con list ---
    tem_ui, tem_extras = _unpack_temcon(result.get("Tem-Con", [None]))

    # Define a new row with the evaluation results
    base_row = {
        "Timestamp": timestamp,
        "Filename": filename,
        "Prompt": prompt,
        "CLIP Score": round(result["clip_score"], 4),
        "PickScore": round(result["pick_score"], 4),
        "Temporal Consistency": (round(tem_ui, 4) if tem_ui is not None else None),
        "Embedding Distance": round(result["embedding_distance"], 4),
        "FPS (Video)": actual_fps,
        "FPS (Eval)": fps,
        "Frames": frame_count,
        "Resolution": resolution,
        "Duration": duration,
    }

    # --- CHANGED: add extra TC columns (will be saved to CSV but not shown in UI table) ---
    # round safely when value is not None
    for k, v in tem_extras.items():
        base_row[k] = (round(v, 4) if isinstance(v, (int, float)) else None)

    new_row = pd.DataFrame([base_row])

    # Add the new row to the history DataFrame
    updated_df = pd.concat([history_df, new_row], ignore_index=True)
    return updated_df, updated_df


def save_table_to_csv(history_df):
    if history_df.empty:
        return None

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_{timestamp}.csv"

    # Define the path to save the CSV file
    temp_dir = tempfile.gettempdir()
    target_path = os.path.join(temp_dir, filename)

    history_df.to_csv(target_path, index=False, encoding="utf-8")
    return target_path


def apply_table_edits(edited_df):
    # Update the history DataFrame with edits made in the Gradio table
    return edited_df


def clear_history():
    empty_df = pd.DataFrame(columns=[
        "Timestamp", "Filename", "Prompt",
        "CLIP Score", "PickScore", "Temporal Consistency",
        "Embedding Distance", "FPS (Video)", "FPS (Eval)",
        "Frames", "Resolution", "Duration",
        "TC per 4f", "TC per 8f", "TC per 12f",
        "TC per 16f", "TC per 20f", "TC per 24f",
    ])
    return empty_df, empty_df


if __name__ == "__main__":
    with gr.Blocks(css="""
        .orange-button {
            background-color: #ff7f0e !important;
            color: white;
            transition: background-color 0.3s ease;
        }
        .orange-button:hover {
            background-color: #e66900 !important;
        }

        .blue-button {
            background-color: #1f77b4 !important;
            color: white;
            transition: background-color 0.3s ease;
        }
        .blue-button:hover {
            background-color: #155d8b !important;
        }

        .green-button {
            background-color: #2ca02c !important;
            color: white;
            transition: background-color 0.3s ease;
        }
        .green-button:hover {
            background-color: #1d7f1d !important;
        }
        .red-button {
            background-color: #d62728 !important;
            color: white;
            transition: background-color 0.3s ease;
        }
        .red-button:hover {
            background-color: #a91d1d !important;
        }
    """) as demo:

        # Title
        gr.Markdown("#  Video Evaluation with Diffusion-Based Metrics")
        
        with gr.Row():
            video_input = gr.Video(label="Upload Video")

            with gr.Column():
                with gr.Group():
                    gr.Markdown("##  Video Info")

                    with gr.Row():
                        video_filename_output = gr.Textbox(label="Filename", interactive=False)
                        video_duration_output = gr.Textbox(label="Duration (sec)", interactive=False)

                    with gr.Row():
                        video_frames_output = gr.Textbox(label="Total Frames", interactive=False)
                        video_fps_output = gr.Textbox(label="FPS", interactive=False)
                        video_resolution_output = gr.Textbox(label="Resolution (width)x(height)", interactive=False)
                
                fps_input = gr.Slider(1, 30, value=8, step=1, label="FPS (for frame extraction)")
        
        prompt_input = gr.Textbox(
            label="Prompt (e.g., 'a bear is walking, anime style')",
            value="a bear is walking, anime style",
            placeholder="Enter the main prompt for evaluation",
            lines=1)
        
        run_button = gr.Button("Run Evaluation", elem_classes="orange-button")
        
        with gr.Row():
            clip_score_output = gr.Textbox(label="CLIP Score (Tex-Ali)")
            pick_score_output = gr.Textbox(label="PickScore")
            temporal_output = gr.Textbox(label="Temporal Consistency (Tem-Con)")
            embedding_output = gr.Textbox(label="Embedding Distance")

        history_df = gr.State(value=pd.DataFrame(columns=[
            "Timestamp", "Filename", "Prompt",
            "CLIP Score", "PickScore", "Temporal Consistency", "Embedding Distance",
            "FPS (Video)", "FPS (Eval)", "Frames", "Resolution", "Duration",
            "TC per 4f", "TC per 8f", "TC per 12f",
            "TC per 16f", "TC per 20f", "TC per 24f",
        ]))

        result_table = gr.Dataframe(
            label="Evaluation History",
            headers=[
                "Timestamp", "Filename", "Prompt",
                "CLIP Score", "PickScore",
                "Temporal Consistency",  
                "TC per 4f", "TC per 8f", "TC per 12f", "TC per 16f", "TC per 20f", "TC per 24f",
                "Embedding Distance", "FPS (Video)", "FPS (Eval)",
                "Frames", "Resolution", "Duration"
            ],
            interactive=True,
            wrap=True
        )

        with gr.Row():
            update_button = gr.Button("Apply Table Edits to History", elem_classes="blue-button")
            clear_button = gr.Button("Clear Table", elem_classes="red-button")
            save_button = gr.Button("Download as CSV", elem_classes="green-button")
            download_file = gr.File(label="Download CSV")

        # Set up event handlers
        run_button.click(
            run_evaluation,
            inputs=[video_input, prompt_input, fps_input],
            outputs=[clip_score_output, pick_score_output, temporal_output, embedding_output]
        )

        run_button.click(
            update_result_table,
            inputs=[video_input, prompt_input, fps_input, history_df],
            outputs=[history_df, result_table]
        )

        update_button.click(
            fn=apply_table_edits,
            inputs=[result_table],
            outputs=[history_df]
        )

        clear_button.click(
            fn=clear_history,
            outputs=[history_df, result_table]
        )

        save_button.click(
            fn=save_table_to_csv,
            inputs=[history_df],
            outputs=[download_file]
        )

        video_input.change(
            update_video_info,
            inputs=video_input,
            outputs=[
                video_input,
                video_filename_output,
                video_duration_output,
                video_frames_output,
                video_fps_output,
                video_resolution_output
            ]
        )

    demo.launch()
