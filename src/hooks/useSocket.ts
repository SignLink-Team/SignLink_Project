import { useEffect, useRef, useState } from "react";


export interface SignResult {
  text: string;
  confidence?: number;
  gesture?: string;
}


export function useSignSocket(
  onResult?: (result: SignResult)=>void
){

  const socketRef = useRef<WebSocket | null>(null);
  const [connected,setConnected] = useState(false);


  useEffect(()=>{

    const ws = new WebSocket(
      "ws://localhost:8000/ws/sign"
    );

    socketRef.current = ws;


    ws.onopen = ()=>{
      console.log("WS connected");
      setConnected(true);
    };


    ws.onmessage = (event)=>{

      const data =
        JSON.parse(event.data);


      onResult?.(data);

    };


    ws.onerror=(e)=>{
      console.error("WS error",e);
    };


    ws.onclose=()=>{
      setConnected(false);
    };


    return ()=>{
      ws.close();
    }

  },[]);

  const sendFrame = (data: Blob | object) => {
    //console.log("sendFrame 호출, readyState:", socketRef.current?.readyState);
    if (
      socketRef.current &&
      socketRef.current.readyState === WebSocket.OPEN
    ) {
      if (data instanceof Blob) {
        socketRef.current.send(data);
      } else {
        socketRef.current.send(JSON.stringify(data));
      }
    } else {
      console.warn("소켓 아직 연결 안 됨:", socketRef.current?.readyState);
    }
  };

  return {
    sendFrame,
    connected
  };

}