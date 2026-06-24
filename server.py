from fastapi import FastAPI, WebSocket
import json
app = FastAPI()


@app.websocket("/ws/sign")
async def sign(ws: WebSocket):
    await ws.accept()
    while True:
        # bytes 대신 text로 받기
        data = await ws.receive_text()
        #print("데이터 수신:", data[:100])
        payload = json.loads(data)
        
        수지 = payload["수지"]    # 261차원 list
        비수지 = payload["비수지"]  # 1404차원 list
        
        # 여기서 모델 추론
        # result = model.predict(수지)
        
        await ws.send_text(json.dumps({
            "text": "번역 결과",
            "confidence": 0.95
        }))