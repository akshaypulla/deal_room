"""
Standalone Gradio UI for DealRoom - Pure Python, no external openenv dependencies.
"""

import gradio as gr
from models import DealRoomAction
from server.deal_room_environment import DealRoomEnvironment
from server.gradio_custom import DealRoomWebManager, build_custom_tab, load_metadata


class DealRoomGradioUI:
    """Gradio UI manager for DealRoom environment."""

    def __init__(self, env: DealRoomEnvironment | None = None):
        self.env = env or DealRoomEnvironment()
        self.current_task_id = "aligned"
        self.current_seed = 42
        self.last_observation = None

    def reset(self, task_id: str, seed: int):
        """Reset the environment."""
        self.current_task_id = task_id
        self.current_seed = seed
        self.last_observation = self.env.reset(seed=seed, task_id=task_id)
        return self._format_observation(self.last_observation)

    def step(self, action_type: str, target: str, message: str):
        """Take a step in the environment."""
        if self.last_observation is None or self.last_observation.done:
            return "Please reset the environment first.", "", self._get_state_info()

        action = DealRoomAction(
            action_type=action_type,
            target=target,
            message=message,
        )
        obs, reward, done, info = self.env.step(action)
        self.last_observation = obs

        response = self._format_response(obs, reward, done, info)
        state_info = self._get_state_info()
        return response, "", state_info

    def get_current_state(self):
        """Get current state."""
        if self.last_observation is None:
            return "Environment not initialized. Please reset."
        return self._get_state_info()

    def _format_observation(self, obs):
        """Format observation for display."""
        lines = [
            f"## Round {obs.round_number}/{obs.max_rounds}",
            f"**Stage:** {obs.deal_stage}",
            f"**Momentum:** {obs.deal_momentum}",
            f"**Days to Deadline:** {obs.days_to_deadline}",
            "",
            "### Stakeholder Messages:",
        ]
        for sid, msg in obs.stakeholder_messages.items():
            if msg:
                lines.append(f"**{sid}:** {msg}")

        if obs.veto_precursors:
            lines.append("")
            lines.append("### Veto Warnings:")
            for sid, warning in obs.veto_precursors.items():
                lines.append(f"⚠️ **{sid}:** {warning}")

        if obs.active_blockers:
            lines.append("")
            lines.append(f"### Active Blockers: {', '.join(obs.active_blockers)}")

        if obs.competitor_events:
            lines.append("")
            lines.append(f"### Competitor Events: {', '.join(obs.competitor_events)}")

        return "\n".join(lines)

    def _format_response(self, obs, reward, done, info):
        """Format step response."""
        lines = [self._format_observation(obs)]

        if done:
            if reward > 0:
                lines.append("")
                lines.append(f"## ✅ Episode Complete! Score: {reward:.4f}")
            else:
                lines.append("")
                lines.append(
                    f"## ❌ Episode Failed: {self.env.state.failure_reason or 'Timeout'}"
                )
        else:
            lines.append("")
            lines.append(f"**Reward:** {reward:.4f}")

        return "\n".join(lines)

    def _get_state_info(self):
        """Get current state info."""
        if self.last_observation is None:
            return "Not initialized"

        state = self.env.state
        return f"""**State:**
- Round: {state.round_number}/{state.max_rounds}
- Stage: {state.deal_stage}
- Deal Closed: {state.deal_closed}
- Deal Failed: {state.deal_failed}
- Failure Reason: {state.failure_reason or "N/A"}
"""


def create_dealroom_gradio_app():
    """Create the Gradio app with Playground and Custom tabs."""
    shared_env = DealRoomEnvironment()
    ui = DealRoomGradioUI(env=shared_env)
    metadata = load_metadata()
    custom_manager = DealRoomWebManager(shared_env, metadata)

    playground_tab = gr.Blocks(title="DealRoom Playground")
    with playground_tab:
        gr.Markdown("# DealRoom Negotiation Environment")
        gr.Markdown(
            "### OpenEnv-compatible multi-stakeholder enterprise negotiation simulator"
        )

        with gr.Row():
            with gr.Column(scale=3):
                task_id = gr.Dropdown(
                    ["aligned", "conflicted", "hostile_acquisition"],
                    value="aligned",
                    label="Scenario",
                    info="Choose the negotiation scenario",
                )
                seed = gr.Number(
                    value=42, label="Seed", info="Random seed for reproducibility"
                )

                with gr.Row():
                    reset_btn = gr.Button("Reset Environment", variant="primary")

                gr.Markdown("### Action Configuration")
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
                    label="Action Type",
                )
                target = gr.Textbox(
                    value="all", label="Target", info="Stakeholder(s) to address"
                )
                message = gr.Textbox(
                    value="Thank you for your time. I appreciate your consideration.",
                    label="Message",
                    lines=3,
                )

                with gr.Row():
                    submit_btn = gr.Button("Submit Action", variant="primary")
                    clear_btn = gr.Button("Clear Message")

            with gr.Column(scale=2):
                output = gr.Markdown("### Output\n*Reset the environment to begin*")
                state_info = gr.Textbox(label="State Info", lines=8, interactive=False)

        # Event handlers
        reset_btn.click(fn=ui.reset, inputs=[task_id, seed], outputs=output)

        submit_btn.click(
            fn=ui.step,
            inputs=[action_type, target, message],
            outputs=[output, message, state_info],
        )

        clear_btn.click(fn=lambda: "", outputs=message)

        # Auto-reset when task changes
        task_id.change(fn=ui.reset, inputs=[task_id, seed], outputs=output)

        gr.Markdown("""
        ---
        ### Usage Instructions:
        1. Select a scenario and seed
        2. Click **Reset Environment** to start
        3. Configure your action (type, target, message)
        4. Click **Submit Action** to take a turn
        5. Monitor stakeholder responses and adjust strategy

        ### Action Types:
        - **direct_message**: Standard communication
        - **send_document**: Share documents (ROI, compliance, etc.)
        - **backchannel**: Informal check-in
        - **group_proposal**: Proposal to all stakeholders
        - **concession**: Offer terms
        - **walkaway_signal**: Signal potential deal failure
        - **reframe_value_prop**: Reframe value proposition
        - **exec_escalation**: Escalate to leadership
        """)

    custom_tab = build_custom_tab(
        custom_manager,
        action_fields=[],
        metadata=metadata,
        is_chat_env=False,
        title=metadata.name,
        quick_start_md=None,
    )

    app = gr.TabbedInterface(
        [playground_tab, custom_tab],
        tab_names=["Playground", "Custom"],
        title="DealRoom",
    )

    return app


def main():
    """Run the Gradio app."""
    app = create_dealroom_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
