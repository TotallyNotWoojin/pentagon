import { error } from "console";
import { NextResponse } from "next/server";
import {put} from "@vercel/blob";
import crypto from "crypto";
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { text } = body;

    const apiSecret=request.headers.get("X-API-SECRET");
    if(apiSecret!==process.env.API_SECRET){
      return NextResponse.json({error: "Unauthorized"}, {status: 401})
    }

    // TODO: Call your Image Generation API here
    // For now, we'll just echo back the text
    const url=new URL("https://faizan--sd-demo-model-generate.modal.run")
    url.searchParams.set("prompt",text)
    const response=await fetch(url.toString(),{method:"GET",headers:{
      "X-API-Key": process.env.API_KEY || "",
      Accept: "image/jpeg",
    },
   });

   if(!response.ok){
    const errorText= await response.text
    console.error("API Response: ",errorText );
    throw new Error(
      `HTTP Error! status: ${response.status}, message: ${errorText}`
    )    
   }


   const imageBuffer=await response.arrayBuffer();
   const filename=`${crypto.randomUUID()}.jpg`
   const blob=await put(filename,imageBuffer,{access: "public", contentType:"image/jpeg",})
    return NextResponse.json({
      success: true,
      imageUrl: blob.url,
    });
  } catch (error) {
    return NextResponse.json(
      { success: false, error: "Failed to process request" },
      { status: 500 }
    );
  }
}