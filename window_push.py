
from winotify import Notification, audio

# 알림 클릭 시 처리할 함수 정의
def on_action_click(action):
    print(f"클릭한 버튼: {action}")

notification = Notification(
    app_id="ROGUN",
    title="""
    로건이의 행동 변화가 감지되었습니다. -----------------------------------------""",
    msg="\" LYING -> WALK \"",
    icon=r"C:\Users\USER\Downloads\pngegg.ico"
)

# # 버튼 추가 및 클릭 시 동작 연결
# notification.add_actions("action_1", "버튼 1", lambda: on_action_click("버튼 1"))
# notification.add_actions("action_2", "버튼 2", lambda: on_action_click("버튼 2"))

# 사운드를 추가
notification.set_audio(audio.LoopingAlarm8, loop=False)

# 버튼 추가 (URL로 이동)
notification.add_actions(label="실시간 영상", launch="https://www.naver.com/")   # 실시간 영상
notification.add_actions(label="행동 기록", launch="https://www.naver.com/")   # 행동 기록


# 알림 보내기
notification.show()




# from winotify import Notification, audio

# # 알림 객체 생성
# notification = Notification(
#     app_id="ROGUN",
#     title="""
#     로건이의 행동 변화가 감지되었습니다. -----------------------------------------""",
#     msg="\" LYING -> WALK \"",
#     icon=r"C:\Users\USER\Downloads\pngegg.ico"
# )

# # 알림을 5초 후에 자동으로 닫히도록 설정
# notification.duration = "short"  # 밀리초 단위로 설정

# # 사운드를 추가
# notification.set_audio(audio.LoopingAlarm8, loop=False)
# # notification.set_audio(r"C:\Users\USER\Downloads\dog.wav", loop=False)

# # 알림 보내기
# notification.show()
