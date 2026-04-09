"""
Standalone Gradio UI for DealRoom - Pure Python, no external openenv dependencies.
"""

from __future__ import annotations

import html
import json

import gradio as gr

from models import DealRoomAction
from server.gradio_custom import DealRoomWebManager, build_custom_tab, load_metadata
from server.session_pool import DealRoomSessionPool

PLAYGROUND_CSS = """
#dealroom-playground-root {
  background: #0d1117;
  color: #e5e7eb;
}
#dealroom-playground-root .playground-shell {
  padding: 4px 4px 10px;
}
#dealroom-playground-root .app-title-block {
  text-align: center;
  padding: 6px 0 10px;
}
#dealroom-playground-root .app-title-block h1 {
  margin: 0;
  color: #f3f4f6;
  font-size: 1.65rem;
  font-weight: 700;
}
#dealroom-playground-root .top-tagline {
  text-align: center;
  margin: 4px 0 0;
  color: #9ca3af;
  font-size: 0.95rem;
}
#dealroom-playground-root .top-divider {
  border-top: 1px solid #2c3440;
  margin: 0 0 12px;
}
#dealroom-playground-root .classic-grid {
  align-items: flex-start;
}
#dealroom-playground-root .classic-panel {
  background: #0f141b;
  border: 1px solid #263241;
  border-radius: 4px;
  padding: 14px;
}
#dealroom-playground-root .classic-panel--playground {
  padding: 0;
  overflow: hidden;
}
#dealroom-playground-root .classic-panel__header {
  background: #464650;
  color: #f3f4f6;
  padding: 12px 14px 10px;
  border-bottom: 1px solid #4d5563;
}
#dealroom-playground-root .classic-panel__header h2 {
  margin: 0 0 4px;
  color: #f3f4f6;
}
#dealroom-playground-root .classic-panel__header p {
  margin: 0;
  color: #ebeef2;
  font-size: 0.95rem;
}
#dealroom-playground-root .classic-panel__body {
  padding: 0;
}
#dealroom-playground-root .classic-section + .classic-section {
  margin-top: 12px;
}
#dealroom-playground-root .classic-panel h2,
#dealroom-playground-root .classic-panel h3 {
  margin-top: 0;
  color: #f3f4f6;
}
#dealroom-playground-root .classic-muted,
#dealroom-playground-root .help-text {
  color: #9ca3af;
}
#dealroom-playground-root .quickstart-copy {
  display: grid;
  gap: 8px;
}
#dealroom-playground-root .quickstart-copy h3 {
  margin: 0 0 4px;
}
#dealroom-playground-root .quickstart-copy p {
  margin: 0;
  color: #e5e7eb;
}
#dealroom-playground-root .snippet-row {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 12px;
  margin-bottom: 6px;
}
#dealroom-playground-root .snippet-title {
  color: #f3f4f6;
  font-size: 0.92rem;
  font-family: "IBM Plex Sans", "Inter", system-ui, sans-serif;
  font-weight: 600;
}
#dealroom-playground-root .quickstart-code {
  margin: 0 0 14px;
}
#dealroom-playground-root .quickstart-code .cm-editor,
#dealroom-playground-root .quickstart-code .cm-scroller,
#dealroom-playground-root .quickstart-code .cm-gutters {
  background: #2f3640 !important;
}
#dealroom-playground-root .quickstart-code .cm-editor {
  border: 1px solid #566170 !important;
  border-radius: 8px !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
}
#dealroom-playground-root .quickstart-code .cm-gutters {
  display: none !important;
}
#dealroom-playground-root .quickstart-code .cm-content,
#dealroom-playground-root .quickstart-code .cm-line {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
  font-size: 0.92rem !important;
}
#dealroom-playground-root .quickstart-code .cm-activeLine,
#dealroom-playground-root .quickstart-code .cm-activeLineGutter {
  background: transparent !important;
}
#dealroom-playground-root .quickstart-link {
  color: #d7dde7;
  text-decoration: underline;
}
#dealroom-playground-root .gr-accordion {
  background: #0f141b !important;
  border: 1px solid #263241 !important;
  border-radius: 4px !important;
}
#dealroom-playground-root .gr-accordion summary,
#dealroom-playground-root .gr-markdown,
#dealroom-playground-root .gr-markdown p,
#dealroom-playground-root label {
  color: #e5e7eb !important;
}
#dealroom-playground-root input,
#dealroom-playground-root textarea,
#dealroom-playground-root select {
  background: #0d1117 !important;
  color: #e5e7eb !important;
  border-color: #2d3948 !important;
}
#dealroom-playground-root .gr-button {
  border-radius: 0 !important;
  box-shadow: none !important;
  min-height: 42px !important;
}
#dealroom-playground-root .gr-button-primary {
  background: #3b4450 !important;
  border-color: #4c5867 !important;
  color: #f3f4f6 !important;
}
#dealroom-playground-root .gr-button-primary:hover {
  background: #434d5a !important;
  border-color: #5b6777 !important;
}
#dealroom-playground-root .gr-button-secondary {
  background: #3b4450 !important;
  border-color: #4c5867 !important;
  color: #f3f4f6 !important;
}
#dealroom-playground-root .gr-form,
#dealroom-playground-root .gr-group,
#dealroom-playground-root .gr-box {
  border-color: #2b3746 !important;
}
#dealroom-playground-root .playground-form {
  padding: 0 0 0;
}
#dealroom-playground-root .playground-output {
  min-height: 140px;
}
#dealroom-playground-root .playground-controls .gradio-row {
  gap: 0 !important;
}
#dealroom-playground-root .playground-controls .gr-button {
  border-left-width: 0 !important;
}
#dealroom-playground-root .playground-controls .gr-button:first-child {
  border-left-width: 1px !important;
}
#dealroom-playground-root .raw-json textarea {
  min-height: 240px !important;
}
"""


class DealRoomGradioUI:
    """Gradio UI manager for DealRoom environment."""

    def __init__(self, pool: DealRoomSessionPool):
        self.pool = pool

    def reset(self, task_id: str, seed: int, session_id: str | None):
        """Reset the environment."""
        session_id, obs, _state = self.pool.reset(
            task_id=task_id, seed=int(seed), session_id=session_id
        )
        return (
            self._format_observation(obs),
            self._get_state_info(session_id),
            session_id,
        )

    def step(self, action_type: str, target: str, message: str, session_id: str | None):
        """Take a step in the environment."""
        if not session_id or not self.pool.has_session(session_id):
            return (
                "⚠️ Please reset the environment first.",
                "",
                "Not initialized",
                session_id,
            )

        action = DealRoomAction(
            action_type=action_type,
            target=target,
            message=message,
        )
        obs, reward, done, info, state = self.pool.step(session_id, action)

        response = self._format_response(
            obs, reward, done, info, state.failure_reason or "Timeout"
        )
        state_info = self._get_state_info(session_id)
        return response, "", state_info, session_id

    def get_current_state(self, session_id: str | None):
        """Get current state."""
        if not session_id or not self.pool.has_session(session_id):
            return "Environment not initialized. Please reset."
        return self._get_state_info(session_id)

    def _format_observation(self, obs):
        """Format observation for display."""
        stage_map = {
            "evaluation": "🎯",
            "negotiation": "📝",
            "legal_review": "⚖️",
            "final_approval": "✅",
            "closed": "🏁",
        }
        stage_icon = stage_map.get(obs.deal_stage, "📋")

        momentum_icons = {"progressing": "⬆️", "stalling": "⏸️", "critical": "🚨"}
        momentum_icon = momentum_icons.get(obs.deal_momentum, "•")

        lines = [
            f"### Round {obs.round_number}/{obs.max_rounds} {stage_icon} {obs.deal_stage.replace('_', ' ').title()}",
            f"{momentum_icon} Momentum: **{obs.deal_momentum.title()}** | ⏱️ {obs.days_to_deadline} days left",
            "",
        ]

        for sid, msg in obs.stakeholder_messages.items():
            if msg:
                lines.append(f"**{sid}:** {msg}")

        if obs.active_blockers:
            blockers = ", ".join([f"🔴 {b}" for b in obs.active_blockers])
            lines.append(f"\n**Blockers:** {blockers}")

        return "\n".join(lines)

    def _format_response(self, obs, reward, done, info, failure_reason: str):
        """Format step response."""
        lines = [self._format_observation(obs)]

        if done:
            if reward > 0:
                lines.append(f"\n## ✅ Deal Closed! Score: **{reward:.2f}**")
            else:
                lines.append(f"\n## ❌ Deal Failed: **{failure_reason}**")
        else:
            lines.append(f"\n**Reward: {reward:.2f}**")

        return "\n".join(lines)

    def _get_state_info(self, session_id: str):
        """Get current state info."""
        state = self.pool.state(session_id)
        return f"""**Round:** {state.round_number}/{state.max_rounds}
**Stage:** {state.deal_stage.replace("_", " ").title()}
**Deal:** {"✅ Closed" if state.deal_closed else "❌ Open"}
**Failed:** {"Yes" if state.deal_failed else "No"}"""


def create_dealroom_gradio_app(pool: DealRoomSessionPool | None = None):
    """Create the Gradio app with Playground and Custom tabs."""
    pool = pool or DealRoomSessionPool()
    ui = DealRoomGradioUI(pool=pool)
    metadata = load_metadata()
    custom_manager = DealRoomWebManager(pool, metadata)

    def snippet_header_html(title: str) -> str:
        return f"""
        <div class="snippet-row">
          <div class="snippet-title">{html.escape(title)}</div>
        </div>
        """

    python_snippet = """from deal_room import DealRoomAction, DealRoomEnvironment

with DealRoomEnvironment() as env:
    obs = env.reset(task_id="aligned", seed=42)
    result = env.step(
        DealRoomAction(message="Help me understand the real blocker.")
    )"""
    server_snippet = 'env = DealRoomEnvironment(base_url="http://localhost:7860")'
    fork_snippet = "openenv fork akshaypulla/deal-room --repo-id <your-username>/deal-room"
    pr_snippet = "cd <forked-repo>\nopenenv push akshaypulla/deal-room --create-pr"

    playground_tab = gr.Blocks(title="DealRoom Playground")
    with playground_tab:
        gr.HTML(f"<style>{PLAYGROUND_CSS}</style>")

        with gr.Row(elem_classes=["classic-grid"]):
            with gr.Column(scale=4):
                with gr.Accordion("Quick Start", open=True):
                    gr.HTML(
                        """
                        <div class="quickstart-copy">
                          <div>
                            <h3>Connect to this environment</h3>
                          </div>
                        </div>
                        """
                    )
                    gr.Markdown(
                        "Connect from Python using `DealRoomEnvironment`:"
                    )
                    gr.HTML(snippet_header_html("Python"))
                    gr.Code(
                        value=python_snippet,
                        language="python",
                        interactive=False,
                        lines=5,
                        show_line_numbers=False,
                        buttons=["copy"],
                        elem_classes=["quickstart-code"],
                        show_label=False,
                    )
                    gr.HTML(
                        """
                        <div class="quickstart-copy">
                          <div>
                            <p>Or connect directly to a running server:</p>
                          </div>
                        </div>
                        """
                    )
                    gr.HTML(snippet_header_html("Direct server mode"))
                    gr.Code(
                        value=server_snippet,
                        language="python",
                        interactive=False,
                        lines=1,
                        show_line_numbers=False,
                        buttons=["copy"],
                        elem_classes=["quickstart-code"],
                        show_label=False,
                    )
                    gr.HTML(
                        """
                        <div class="quickstart-copy">
                          <div>
                            <h3>Contribute to this environment</h3>
                            <p>Submit improvements via pull request on the Hugging Face Hub.</p>
                          </div>
                        </div>
                        """
                    )
                    gr.HTML(snippet_header_html("Fork this environment"))
                    gr.Code(
                        value=fork_snippet,
                        language="shell",
                        interactive=False,
                        lines=1,
                        show_line_numbers=False,
                        buttons=["copy"],
                        elem_classes=["quickstart-code"],
                        show_label=False,
                    )
                    gr.HTML(
                        """
                        <div class="quickstart-copy">
                          <p>Then make your changes and submit a pull request:</p>
                        </div>
                        """
                    )
                    gr.HTML(snippet_header_html("Submit your PR"))
                    gr.Code(
                        value=pr_snippet,
                        language="shell",
                        interactive=False,
                        lines=2,
                        show_line_numbers=False,
                        buttons=["copy"],
                        elem_classes=["quickstart-code"],
                        show_label=False,
                    )
                    gr.HTML(
                        """
                        <div class="quickstart-copy">
                          <p>For more information, see the <a class="quickstart-link" href="https://github.com/huggingface/openenv" target="_blank">OpenEnv documentation</a>.</p>
                        </div>
                        """
                    )
                with gr.Accordion("README", open=False):
                    gr.Markdown(metadata.readme_content or "README not available.")

            with gr.Column(scale=8):
                with gr.Group(elem_classes=["classic-panel", "classic-panel--playground"]):
                    gr.HTML(
                        """
                        <div class="classic-panel__header">
                          <h2>Playground</h2>
                          <p>Click Reset to start a new episode.</p>
                        </div>
                        """
                    )
                    with gr.Column(elem_classes=["classic-panel__body", "playground-form"]):
                        with gr.Row():
                            task_id = gr.Dropdown(
                                ["aligned", "conflicted", "hostile_acquisition"],
                                value="aligned",
                                label="Scenario",
                                info="Choose the negotiation scenario",
                            )
                            seed = gr.Number(
                                value=42,
                                label="Seed",
                                precision=0,
                                placeholder="42",
                            )

                        action_type = gr.Dropdown(
                            [
                                "direct_message",
                                "send_document",
                                "backchannel",
                                "group_proposal",
                                "concession",
                                "walkaway_signal",
                                "reframe_value_prop",
                                "exec_escalation",
                            ],
                            value="direct_message",
                            label="Move Type",
                        )
                        target = gr.Textbox(
                            value="all",
                            label="Target",
                            placeholder="all",
                        )
                        terms = gr.Textbox(
                            value="",
                            label="Terms",
                            lines=3,
                            placeholder='{"price": 180000, "timeline_weeks": 14, "support_level": "named_support_lead"}',
                        )
                        message = gr.Textbox(
                            value="Help me understand the real approval concern we still need to address.",
                            label="Message",
                            lines=4,
                            placeholder="Enter message...",
                        )

                        with gr.Row(elem_classes=["playground-controls"]):
                            submit_btn = gr.Button("Step", variant="primary")
                            reset_btn = gr.Button("Reset", variant="primary")
                            getstate_btn = gr.Button("Get state", variant="primary")

                        output = gr.Markdown("### Status\nReset the environment to begin", elem_classes=["playground-output"])
                        raw_json = gr.Textbox(
                            label="Raw JSON response",
                            lines=18,
                            interactive=False,
                            placeholder="{}",
                            elem_classes=["raw-json"],
                        )
                        state_info = gr.Textbox(
                            label="State Info",
                            lines=8,
                            interactive=False,
                        )

        session_state = gr.State(None)

        def parse_terms(raw_terms: str):
            raw_terms = (raw_terms or "").strip()
            if not raw_terms:
                return None, None
            try:
                return json.loads(raw_terms), None
            except json.JSONDecodeError as exc:
                return None, f"Invalid terms JSON: {exc}"

        def reset_fn(task, s):
            session_id, obs, _state = pool.reset(task_id=task, seed=int(s), session_id=None)
            raw = {
                "observation": obs.model_dump(),
                "state": pool.state(session_id).model_dump(),
                "session_id": session_id,
            }
            status = ui._format_observation(obs)
            state_block = ui._get_state_info(session_id)
            return status, json.dumps(raw, indent=2), state_block, session_id

        def step_fn(action, tgt, raw_terms, msg, sid):
            if not sid or not pool.has_session(sid):
                return (
                    "### Status\n⚠️ Please reset the environment first.",
                    json.dumps({"error": "not_initialized"}, indent=2),
                    "Not initialized",
                    sid,
                )
            proposed_terms, terms_error = parse_terms(raw_terms)
            if terms_error:
                return (
                    f"### Status\n⚠️ {terms_error}",
                    json.dumps({"error": terms_error}, indent=2),
                    ui._get_state_info(sid),
                    sid,
                )
            action_payload = DealRoomAction(
                action_type=action,
                target=tgt,
                message=msg,
                proposed_terms=proposed_terms,
            )
            obs, reward, done, info, state = pool.step(sid, action_payload)
            status = ui._format_response(obs, reward, done, info, state.failure_reason or "Timeout")
            raw = {
                "observation": obs.model_dump(),
                "reward": reward,
                "done": done,
                "info": info,
                "state": state.model_dump(),
            }
            return status, json.dumps(raw, indent=2), ui._get_state_info(sid), sid

        def getstate_fn(sid):
            if not sid or not pool.has_session(sid):
                return "Environment not initialized. Please reset.", json.dumps({"error": "not_initialized"}, indent=2)
            state = pool.state(sid).model_dump()
            return ui._get_state_info(sid), json.dumps(state, indent=2)

        reset_btn.click(
            fn=reset_fn,
            inputs=[task_id, seed],
            outputs=[output, raw_json, state_info, session_state],
        )

        submit_btn.click(
            fn=step_fn,
            inputs=[action_type, target, terms, message, session_state],
            outputs=[output, raw_json, state_info, session_state],
        )
        getstate_btn.click(
            fn=getstate_fn,
            inputs=[session_state],
            outputs=[state_info, raw_json],
        )

    custom_tab = build_custom_tab(
        custom_manager,
        action_fields=[],
        metadata=metadata,
        is_chat_env=False,
        title=metadata.name,
        quick_start_md=None,
    )

    app = gr.Blocks(title="DealRoom", elem_id="dealroom-playground-root")
    with app:
        gr.HTML(f"<style>{PLAYGROUND_CSS}</style>")
        gr.HTML(
            """
            <div class="playground-shell">
              <div class="app-title-block">
                <h1>OpenEnv Agentic Environment: DealRoom</h1>
                <div class="top-tagline">Classic Playground for the multi-stakeholder enterprise negotiation simulator.</div>
              </div>
              <div class="top-divider"></div>
            </div>
            """
        )
        with gr.Tabs():
            with gr.Tab("Playground"):
                playground_tab.render()
            with gr.Tab("Custom"):
                custom_tab.render()

    return app


def main():
    """Run the Gradio app."""
    app = create_dealroom_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
