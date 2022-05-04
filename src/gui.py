import streamlit as st


emojis = {
    "up": "🔼",
    "down": "🔽",
    "left": "◀",
    "right": "▶",
    "up": "🔼",
    "down": "🔽",
    "clockwise": "⏩",
    "counterclockwise": "⏪",
    "backward": "🔴",
    "forward": "🟢",
}

def main():

    st.title("Tello Drone Navigation")

    with st.sidebar:
        st.subheader("User Inputs")

        mode = st.radio("Mode", ["Keyboard Controller", "Gesture Controller"])
        # st.markdown("-----------")

        st.markdown("`On/Off`")
        l, r = st.columns(2)
        land = l.button("Land Drone")
        takeoff = r.button("Takeoff Drone")
        st.markdown("-----------")

        cmds = {}
        if mode == "Keyboard Controller":
            st.markdown("`Keyboard Controller`")
            rows = [st.columns(4), st.columns(4)]
            for i, command in enumerate(emojis.keys()):
                cmds.update({
                    command: rows[i % 2][i // 2].button(emojis[command])
                })
        else:
            st.subheader("Gesture Controller from Webcam")




if __name__ == "__main__":
    main()