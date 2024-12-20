"use server";

import  ImageGenerator from "./components/imageGenerator";
import { generateImage } from "./actions/generateImage";

export default async function Home() {
  return <ImageGenerator generateImage={generateImage} />;

  
}