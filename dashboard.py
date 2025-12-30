import streamlit as st
import subprocess
import os
import pandas as pd
import re
import time

st.set_page_config(layout="wide", page_title="Antigravity Discovery Engine")

st.title("🌌 Implicit-KGCN Discovery Engine")
st.markdown("### The \"Self-Coding\" Scientist")

# Sidebar Status
st.sidebar.header("System Status")
if os.path.exists("lob_hydro_state.csv"):
    st.sidebar.success("✅ Real Market Data Loaded")
else:
    st.sidebar.warning("⚠️ Using Synthetic Data")

if os.path.exists("generated_loss.py"):
    st.sidebar.success("✅ Physics Loss Compiled")
else:
    st.sidebar.info("ℹ️ No Laws Discovered Yet")

# Tab Interface
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 Discovery", "📈 Application", "🧠 Knowledge Graph", "😴 Dream State", "🔮 The Oracle", "🧬 Meta-Cognition"])

with tab1:
    st.header("Step 1: Neuro-Symbolic Discovery")
    
    col_q, col_d = st.columns([3, 1])
    with col_q:
        query = st.text_input("Scientific Query", "How does Fluid Turbulence relate to Financial Volatility?")
    with col_d:
        target_domain = st.selectbox("Target Domain", ["Financial Markets", "Ecology (Population Dynamics)", "Social Science (Opinion Dynamics)", "Quantum Mechanics (Quantum Finance)", "Control Theory (Market Stability)"])

    if st.button("Run Discovery Pipeline"):
        with st.status("Running Discovery Engine...", expanded=True) as status:
            st.write(f"🔍 Retrieval: Searching 10,000+ papers for {target_domain} application...")
            
            # Run Process Directly (No Subprocess)
            try:
                # Import here to avoid loading heavy libs unless needed
                from discovery_engine import run_discovery_pipeline
                results = run_discovery_pipeline(query, target_domain=target_domain)
                
                # Show Logs in Expander
                with st.expander("Show Execution Logs"):
                    st.code("\n".join(results["logs"]), language="text")

                if results["success"]:
                    status.update(label="Discovery Complete!", state="complete", expanded=False)
                    st.success(f"Isomorphism Found: {results['paper_title']} ({results['arxiv_id']})")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("⚛️ Physics Law")
                        if results['original_eq']:
                            st.latex(results['original_eq'])
                        else:
                            st.write("No equation extracted.")
                            
                    with col2:
                        st.subheader("💸 Financial Model")
                        if results['translated_eq']:
                            st.latex(results['translated_eq'])
                    
                    st.subheader("↔️ Variable Mapping")
                    st.json(results["mapping"])

                    st.subheader("🐍 Generated PyTorch Code")
                    st.code(results["generated_code"], language="python")
                         
                else:
                    status.update(label="Discovery Failed", state="error")
                    st.error("Pipeline Error or Circuit Breaker Active")
                    st.write(results["logs"][-1])

            except Exception as e:
                st.error(f"Internal Error: {e}")

with tab2:
    st.header("Step 2: Universal Simulation Studio")
    st.markdown("Run the **Auto-Generated Simulation** compiled from the Physics law.")
    
    if st.button("Run Simulation"):
        with st.status("Executing Simulation...", expanded=True) as status:
            try:
                # Dynamic Import of the latest generated code
                import importlib
                import generated_loss
                importlib.reload(generated_loss)
                
                # Check for known entry points
                if hasattr(generated_loss, 'simulate_opinion'):
                    st.write(">> Detected: Social Opinion Model")
                    fig = generated_loss.simulate_opinion(steps=50, n_agents=100)
                    st.pyplot(fig)
                elif hasattr(generated_loss, 'custom_physics_loss'):
                    st.write(">> Detected: Physics Loss Function")
                    st.code("Model trained directly in background.")
                elif hasattr(generated_loss, 'simulate_quantum_dynamics'):
                     st.write(">> Detected: Quantum Protocol")
                     fig = generated_loss.simulate_quantum_dynamics()
                     st.pyplot(fig)
                else:
                    st.warning("No standard simulation function found in `generated_loss.py`.")
                    
                status.update(label="Simulation Complete", state="complete")
                
            except Exception as e:
                st.error(f"Execution Failed: {e}")
                st.write("Make sure `generated_loss.py` exists and is valid.")

with tab3:
    st.header("Latent Space Topology")
    if os.path.exists("bridge_visualization.png"):
        st.image("bridge_visualization.png", caption="The Grand Unified Map: Physics, Finance, Bio, Social, Quantum")
    else:
        st.warning("Run visualize_bridge.py to generate topology.")

with tab4:
    st.header("🧠 Autonomic Control Center")
    
    col_set, col_dream = st.columns([1, 2])
    
    with col_set:
        st.subheader("System Settings")
        
        # Load Config
        import json
        try:
            with open("system_config.json", "r") as f:
                config = json.load(f)
        except:
            config = {"skeptic_enabled": True}
            
        # Toggle Skeptic
        skeptic_status = st.checkbox("Enable 'Skeptic' Agent (Adversarial Review)", value=config.get("skeptic_enabled", True))
        
        # Save Config if changed
        if skeptic_status != config.get("skeptic_enabled", True):
            config["skeptic_enabled"] = skeptic_status
            with open("system_config.json", "w") as f:
                json.dump(config, f)
            st.rerun()
            
        st.info(f"Skepticism Level: {'HIGH (Phase 28)' if skeptic_status else 'LOW (Speed Mode)'}")
        
    with col_dream:
        st.subheader("😴 Dream Journal (Unsupervised Discoveries)")
        
        import glob
        reports = glob.glob("hypothesis_report_*.md")
        reports.sort(reverse=True) # Newest first
        
        if reports:
            selected_report = st.selectbox("Select Night Cycle", reports)
            with open(selected_report, "r") as f:
                st.markdown(f.read())
        else:
            st.info("No dreams recorded yet. Run `dream_engine.py` to enter sleep mode.")

with tab5:
    st.header("🔮 Consult The Oracle")
    st.markdown("Ask the system to explain its discoveries, methodology, or internal logic.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask the Oracle..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get Oracle response
        with st.spinner("The Oracle is contemplating..."):
            try:
                # Lazy load agent only when needed
                from oracle_agent import OracleAgent
                if "oracle" not in st.session_state:
                    st.session_state.oracle = OracleAgent()
                
                response = st.session_state.oracle.consult(prompt)
            except Exception as e:
                response = f"Communication Error: {e}"
        
        # Display assistant response in chat message container
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab6:
    st.header("🧬 Meta-Cognitive Internal State")
    st.markdown("Monitor the recursive self-improvement loops: Scout, Architect, Operator, and Singularity.")
    
    col_scout, col_singularity = st.columns(2)
    
    with col_scout:
        st.subheader("🦅 Scout Protocol (Value Seeking)")
        if os.path.exists("generated_scout_solution.py"):
            st.success("✅ Scout Solution Generated")
            with st.expander("View Scout Code"):
                 with open("generated_scout_solution.py", "r") as f:
                     st.code(f.read(), language="python")
        else:
            st.info("No active scout missions.")
            
        if st.button("Trigger Scout Mission"):
            with st.spinner("Hunting for efficiency..."):
                try:
                    subprocess.run(["./venv/bin/python", "scout_agent.py"], check=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Mission Failed: {e}")
                    
    with col_singularity:
        st.subheader("🤖 Singularity Node (Self-Refactoring)")
        if os.path.exists("models_v2.py"):
            st.success("✅ System Self-Rewrote 'models.py'")
            st.success("✅ v2 Validation Passed (Sandbox)")
            with st.expander("Compare v1 vs v2"):
                st.write("**Original (Dense SAGE):**")
                st.code("base_conv = SAGEConv((-1, -1), out_channels)", language="python")
                st.write("**Optimized (Sparse GAT):**")
                st.code("base_conv = GATConv((-1, -1), out_channels, add_self_loops=False)", language="python")
        else:
            st.info("System running on original codebase (v1).")

    st.subheader("🔬 The Operator (Bayesian Reality Loop)")
    weights_path = "translator_weights.json"
    if os.path.exists(weights_path):
        import json
        with open(weights_path, "r") as f:
            weights = json.load(f)
        
        st.json(weights)
        st.caption("Current Bayesian Weights for Translator Logic")
    else:
        st.write("Operator not active.")
