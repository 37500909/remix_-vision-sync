import React, { useEffect, useRef, useState } from 'react';
import { GoogleGenAI } from '@google/genai';
import { Camera, Square, Play, Music, Loader2, AlertCircle, Key, Activity, Cpu, ScanFace, Info, X } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { motion, AnimatePresence } from 'motion/react';
import * as Tone from 'tone';

let hoverSynth: Tone.Synth | null = null;

async function initAudio() {
  if (Tone.context.state !== 'running') {
    await Tone.start().catch(() => {});
  }
  if (!hoverSynth) {
    hoverSynth = new Tone.Synth({
      oscillator: { type: 'sine' },
      envelope: { attack: 0.01, decay: 0.1, sustain: 0, release: 0.01 }
    }).toDestination();
    hoverSynth.volume.value = -15;
  }
}

declare global {
  interface Window {
    aistudio?: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

interface SmoothedBox {
  x: number;
  y: number;
  width: number;
  height: number;
  class: string;
  score: number;
  opacity: number;
  labelX: number;
  labelY: number;
}

class PCMPlayer {
  audioContext: AudioContext;
  nextStartTime: number;

  constructor(sampleRate: number = 48000) {
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate });
    this.nextStartTime = this.audioContext.currentTime;
  }

  playChunk(base64Data: string) {
    const binaryString = atob(base64Data);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    
    // 16-bit PCM stereo
    const int16Array = new Int16Array(bytes.buffer);
    const numSamples = int16Array.length / 2;
    const leftChannel = new Float32Array(numSamples);
    const rightChannel = new Float32Array(numSamples);
    
    for (let i = 0; i < numSamples; i++) {
      leftChannel[i] = int16Array[i * 2] / 32768.0;
      rightChannel[i] = int16Array[i * 2 + 1] / 32768.0;
    }

    const audioBuffer = this.audioContext.createBuffer(2, numSamples, this.audioContext.sampleRate);
    audioBuffer.getChannelData(0).set(leftChannel);
    audioBuffer.getChannelData(1).set(rightChannel);

    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);

    const currentTime = this.audioContext.currentTime;
    if (this.nextStartTime < currentTime) {
      this.nextStartTime = currentTime + 0.05;
    }

    source.start(this.nextStartTime);
    this.nextStartTime += audioBuffer.duration;
  }

  stop() {
    if (this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
  }
}

class ProceduralMusicEngine {
  audioContext: AudioContext;
  isPlaying: boolean = false;
  currentVibe: string = 'minimalist ambient drone, quiet';
  targetVibe: string = 'minimalist ambient drone, quiet';
  vibeBlend: number = 1.0;
  nextNoteTime: number = 0;
  timerID: number | null = null;
  
  // Scales (intervals from root)
  scales: Record<string, number[]> = {
    major: [0, 2, 4, 5, 7, 9, 11],
    minor: [0, 2, 3, 5, 7, 8, 10],
    pentatonic: [0, 2, 4, 7, 9],
    cyberpunk: [0, 3, 7, 8, 10], // Phrygian dominant-ish
    drone: [0, 7], // Just roots and fifths
    melancholic: [0, 2, 3, 7, 8], // Minor pentatonic-ish
    dissonant: [0, 1, 6, 7, 11], // For fear/disgust
    tribal: [0, 3, 5, 7, 10] // Minor pentatonic
  };

  constructor() {
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
  }

  setVibe(vibe: string) {
    if (this.targetVibe !== vibe) {
      if (this.vibeBlend >= 1.0) {
        this.currentVibe = this.targetVibe;
      }
      this.targetVibe = vibe;
      this.vibeBlend = 0.0;
    }
  }

  start() {
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
    this.isPlaying = true;
    this.nextNoteTime = this.audioContext.currentTime + 0.1;
    this.scheduleNext();
  }

  stop() {
    this.isPlaying = false;
    if (this.timerID !== null) {
      clearTimeout(this.timerID);
      this.timerID = null;
    }
    if (this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
  }

  playNote(freq: number, type: OscillatorType, duration: number, vol: number, attack: number, time: number) {
    if (this.audioContext.state === 'closed') return;
    
    // Create multiple oscillators for a thicker soundscape
    const numOscs = 4;
    const masterGain = this.audioContext.createGain();
    masterGain.connect(this.audioContext.destination);
    
    const now = time;
    masterGain.gain.setValueAtTime(0, now);
    masterGain.gain.linearRampToValueAtTime(vol, now + attack);
    masterGain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    // Add a subtle reverb effect using a convolver or just delay
    const delay = this.audioContext.createDelay();
    delay.delayTime.value = 0.33;
    const feedback = this.audioContext.createGain();
    feedback.gain.value = 0.4;
    delay.connect(feedback);
    feedback.connect(delay);
    delay.connect(masterGain);

    for (let i = 0; i < numOscs; i++) {
      const osc = this.audioContext.createOscillator();
      const filter = this.audioContext.createBiquadFilter();
      
      osc.type = i % 2 === 0 ? type : 'sine';
      osc.frequency.value = freq * (1 + (i * 0.008)); // Slight detune
      
      filter.type = 'lowpass';
      filter.frequency.value = freq * 2;
      filter.frequency.linearRampToValueAtTime(freq * 6, now + attack);
      filter.frequency.linearRampToValueAtTime(freq * 1.5, now + duration);
      
      osc.connect(filter);
      filter.connect(masterGain);
      filter.connect(delay); // Send to delay for space
      
      osc.start(now);
      osc.stop(now + duration);
    }
  }

  getTempoForVibe(vibe: string): number {
    if (vibe.includes('tribal') || vibe.includes('rhythmic')) return 100;
    if (vibe.includes('cyberpunk') || vibe.includes('electronic')) return 60;
    return 40;
  }

  scheduleNext() {
    if (!this.isPlaying) return;
    
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
    
    while (this.nextNoteTime < this.audioContext.currentTime + 0.5) {
      if (this.vibeBlend < 1.0) {
        this.vibeBlend += 0.02; // crossfade over 50 notes for a much smoother transition
        if (this.vibeBlend > 1.0) this.vibeBlend = 1.0;
      }

      if (this.vibeBlend < 1.0) {
        // Equal power crossfade for smoother audio blending
        const currentWeight = Math.cos(this.vibeBlend * 0.5 * Math.PI);
        const targetWeight = Math.sin(this.vibeBlend * 0.5 * Math.PI);
        this.generateTickForVibe(this.currentVibe, currentWeight, this.nextNoteTime);
        this.generateTickForVibe(this.targetVibe, targetWeight, this.nextNoteTime);
      } else {
        this.generateTickForVibe(this.targetVibe, 1.0, this.nextNoteTime);
      }
      
      // Smoothly interpolate tempo
      const currentTempo = this.getTempoForVibe(this.currentVibe);
      const targetTempo = this.getTempoForVibe(this.targetVibe);
      const tempo = currentTempo * (1 - this.vibeBlend) + targetTempo * this.vibeBlend;
      
      const secondsPerBeat = 60.0 / tempo;
      this.nextNoteTime += secondsPerBeat; // Quarter notes
    }
    
    this.timerID = window.setTimeout(() => this.scheduleNext(), 50);
  }

  generateTickForVibe(vibe: string, weight: number, time: number) {
    if (weight <= 0.01) return;
    
    const isCyberpunk = vibe.includes('cyberpunk') || vibe.includes('electronic');
    const isTribal = vibe.includes('tribal') || vibe.includes('rhythmic') || vibe.includes('happy');
    const isAcoustic = vibe.includes('acoustic') || vibe.includes('guitar');
    const isAmbient = vibe.includes('ambient') || vibe.includes('drone');
    const isSad = vibe.includes('sad') || vibe.includes('melancholy');
    const isTense = vibe.includes('angry') || vibe.includes('fear') || vibe.includes('disgust');
    
    let scale = this.scales.pentatonic;
    let baseNote = 48; // C3
    let oscType: OscillatorType = 'sine';
    let vol = 0.08;
    let duration = 6.0; // Longer durations for soundscape
    let attack = 3.0;

    if (isCyberpunk) {
      scale = this.scales.cyberpunk;
      baseNote = 36; // C2
      oscType = 'sawtooth';
      vol = 0.04;
      duration = 4.0;
      attack = 2.0;
    } else if (isTribal) {
      scale = this.scales.tribal;
      baseNote = 43; // G2
      oscType = 'square';
      vol = 0.06;
      duration = 1.5;
      attack = 0.1;
    } else if (isSad) {
      scale = this.scales.melancholic;
      baseNote = 48;
      oscType = 'sine';
      vol = 0.08;
      duration = 8.0;
      attack = 4.0;
    } else if (isTense) {
      scale = this.scales.dissonant;
      baseNote = 36;
      oscType = 'sawtooth';
      vol = 0.05;
      duration = 5.0;
      attack = 1.5;
    } else if (isAcoustic) {
      scale = this.scales.major;
      baseNote = 48;
      oscType = 'sine';
      vol = 0.08;
      duration = 5.0;
      attack = 2.0;
    } else if (isAmbient) {
      scale = this.scales.drone;
      baseNote = 36;
      oscType = 'sine';
      vol = 0.12;
      duration = 10.0;
      attack = 5.0;
    }

    vol *= weight; // Apply crossfade weight

    // Randomly play a note from the scale
    if (Math.random() > 0.2) {
      const noteIndex = scale[Math.floor(Math.random() * scale.length)];
      const freq = 440 * Math.pow(2, (baseNote + noteIndex - 69) / 12);
      this.playNote(freq, oscType, duration, vol, attack, time);
    }
    
    // Add a bass drone
    if (Math.random() > 0.5) {
      const bassFreq = 440 * Math.pow(2, (baseNote - 12 - 69) / 12);
      this.playNote(bassFreq, 'sine', duration * 2, vol * 1.5, attack * 2, time);
    }
  }
}

const VIBE_MAP: Record<string, string> = {
  person: "zumbido ambiental etéreo, tranquilo",
  'cell phone': "synthwave cyberpunk, electrónico",
  laptop: "synthwave cyberpunk, electrónico",
  tv: "synthwave cyberpunk, electrónico",
  cup: "jazz de cafetería, acústico relajado",
  bottle: "jazz de cafetería, acústico relajado",
  bowl: "jazz de cafetería, acústico relajado",
  cat: "guitarra acústica juguetona, melodía alegre",
  dog: "guitarra acústica juguetona, melodía alegre",
  bird: "guitarra acústica juguetona, melodía alegre",
  car: "ritmo de rock dinámico, tempo rápido",
  bus: "ritmo de rock dinámico, tempo rápido",
  truck: "ritmo de rock dinámico, tempo rápido",
  chair: "zumbido ambiental, relajante",
  couch: "zumbido ambiental, relajante",
  bed: "zumbido ambiental, relajante",
  'potted plant': "flauta etérea, naturaleza ambiental",
  book: "piano clásico, enfocado",
};

function getVibeForObjects(objects: string[]) {
  if (objects.length === 0) return "zumbido ambiental minimalista, silencioso";
  
  const vibes = new Set<string>();
  for (const obj of objects) {
    if (VIBE_MAP[obj]) {
      vibes.add(VIBE_MAP[obj]);
    } else {
      vibes.add("ritmo lofi relajado");
    }
  }
  
  return Array.from(vibes).slice(0, 2).join(", ");
}

const getVibeFromGemini = async (objects: string[], emotion: string, userIdentity: string = 'User'): Promise<string> => {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || process.env.API_KEY });
    const nameStr = userIdentity !== 'Unknown' ? `llamada ${userIdentity}` : 'desconocida';
    const SpanishEmotion = translateEmotion(emotion);
    const prompt = `Eres un generador de paisajes sonoros. Basándote en la siguiente escena, genera una descripción de paisaje sonoro ambiental de 3 a 5 palabras en español (por ejemplo, 'zumbido rítmico tribal', 'zumbido electrónico cyberpunk' o 'ambiente acústico melancólico'). No incluyas ningún otro texto. Nunca devuelvas 'pop', 'alegre' o 'enérgico'. Todo debe ser ambiental. Escena: Una persona ${nameStr} se siente ${SpanishEmotion} y los siguientes objetos son visibles: ${objects.length > 0 ? objects.join(', ') : 'ninguno'}.`;
    const response = await ai.models.generateContent({
      model: 'gemini-flash-lite-latest',
      contents: prompt,
    });
    return response.text?.trim() || "zumbido ambiental, relajante";
  } catch (e: any) {
    console.warn("Gemini API error (falling back to local vibe map):", e.message || e);
    const nameStr = userIdentity !== 'Unknown' ? `usuario ${userIdentity}` : 'usuario';
    const SpanishEmotion = translateEmotion(emotion);
    return getVibeForObjects(objects) + `, estado de ánimo ${SpanishEmotion} del ${nameStr}`;
  }
};

function seedRandom(seedStr: string) {
  let h = 0;
  for (let i = 0; i < seedStr.length; i++) {
    h = (h << 5) - h + seedStr.charCodeAt(i);
    h |= 0;
  }
  return function() {
    h = (h * 1664525 + 1013904223) | 0;
    return (h >>> 0) / 4294967296;
  };
}

interface ConstellationPoint {
  x: number;
  y: number;
  size: number;
}

function getConstellationPoints(identity: string, timeSeed: number = 0): ConstellationPoint[] {
  // If identity is Unknown, use a dynamic jitter seed so the constellation moves unstable.
  // If identity is recognized, lock it with the user's name as seed.
  const seed = identity === 'Unknown' ? `unknown_${Math.floor(timeSeed / 250)}` : identity;
  const rand = seedRandom(seed);
  const points: ConstellationPoint[] = [];
  
  const zones = [
    // Oval contour (12 points)
    ...Array.from({ length: 12 }, (_, i) => {
      const angle = Math.PI * (i / 11) + Math.PI * 0;
      return { cx: Math.cos(angle) * 45, cy: Math.sin(angle) * 55, r: 6 };
    }),
    // Left eye (4 points)
    { cx: -18, cy: -15, r: 4 },
    { cx: -12, cy: -15, r: 4 },
    { cx: -15, cy: -18, r: 4 },
    { cx: -15, cy: -12, r: 4 },
    // Right eye (4 points)
    { cx: 18, cy: -15, r: 4 },
    { cx: 12, cy: -15, r: 4 },
    { cx: 15, cy: -18, r: 4 },
    { cx: 15, cy: -12, r: 4 },
    // Nose bridge and tip (5 points)
    { cx: 0, cy: -10, r: 4 },
    { cx: 0, cy: 0, r: 4 },
    { cx: 0, cy: 10, r: 4 },
    { cx: -8, cy: 10, r: 4 },
    { cx: 8, cy: 10, r: 4 },
    // Mouth (8 points)
    ...Array.from({ length: 8 }, (_, i) => {
      const angle = Math.PI * 2 * (i / 8);
      return { cx: Math.cos(angle) * 15, cy: 25 + Math.sin(angle) * 8, r: 4 };
    })
  ];
  
  zones.forEach((z) => {
    const offsetX = (rand() - 0.5) * z.r * 1.5;
    const offsetY = (rand() - 0.5) * z.r * 1.5;
    points.push({
      x: z.cx + offsetX,
      y: z.cy + offsetY,
      size: 1 + rand() * 1.5
    });
  });
  
  return points;
}

const translateEmotion = (emotion: string): string => {
  const map: Record<string, string> = {
    'happy': 'feliz',
    'sadness': 'tristeza',
    'sad': 'tristeza',
    'surprised': 'sorprendido',
    'surprise': 'sorprendido',
    'angry': 'enojado',
    'fear': 'miedo',
    'disgust': 'disgustado',
    'neutral': 'neutral'
  };
  return map[emotion.toLowerCase()] || emotion;
};

export default function App() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [status, setStatus] = useState('Cargando modelo de detección de objetos...');
  const [currentPrompt, setCurrentPrompt] = useState('Esperando cámara...');
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [infoMsg, setInfoMsg] = useState<string | null>(null);
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const [consoleState, setConsoleState] = useState({
    emotion: 'neutral',
    objects: [] as string[],
    blendshapes: { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 }
  });

  // DeepFace-specific states & refs
  const [analysisEngine, setAnalysisEngine] = useState<'mediapipe' | 'deepface'>('mediapipe');
  const [deepFaceStatus, setDeepFaceStatus] = useState<'offline' | 'connecting' | 'connected'>('offline');
  const [demographics, setDemographics] = useState<{ age: number; gender: string } | null>(null);

  // Biometric Auth & Registration
  const [identity, setIdentity] = useState<string>('Unknown');
  const [isFaceDetected, setIsFaceDetected] = useState<boolean>(false);
  const [loggedInUser, setLoggedInUser] = useState<string | null>(null);
  const [isRegistering, setIsRegistering] = useState<boolean>(false);
  const [firstName, setFirstName] = useState<string>('');
  const [lastName, setLastName] = useState<string>('');
  const [dni, setDni] = useState<string>('');
  const [loggedInDni, setLoggedInDni] = useState<string | null>(null);
  const [registrationProgress, setRegistrationProgress] = useState<number | null>(null);
  const [registrationStatusText, setRegistrationStatusText] = useState<string>('');
  
  // Interactive Biometric Quality States & Refs
  const [enrollmentStep, setEnrollmentStep] = useState<'idle' | 'liveness' | 'frontal' | 'left' | 'right' | 'submitting'>('idle');
  const enrollmentStepRef = useRef<'idle' | 'liveness' | 'frontal' | 'left' | 'right' | 'submitting'>('idle');
  const enrollmentPhotosRef = useRef<string[]>([]);
  const [hudPose, setHudPose] = useState({ yaw: 0, roll: 0, pitch: 0 });
  const [hudQuality, setHudQuality] = useState({ brightness: 120, isNeutral: true, warning: '' });
  const [accessLogs, setAccessLogs] = useState<any[]>([]);
  const [activeUsers, setActiveUsers] = useState<{ name: string; dni: string }[]>([]);
  const [detectedFaces, setDetectedFaces] = useState<any[]>([]);
  const [selectedFace, setSelectedFace] = useState<{ x: number; y: number; w: number; h: number; image: string } | null>(null);

  const fetchLogs = async () => {
    try {
      const res = await fetch('http://localhost:5000/logs');
      const data = await res.json();
      if (data.logs) {
        setAccessLogs(data.logs);
      }
    } catch (err) {
      console.warn("Failed to fetch access logs:", err);
    }
  };

  const analysisEngineRef = useRef<'mediapipe' | 'deepface'>('mediapipe');
  const deepFaceEmotionRef = useRef<string>('neutral');
  const identityRef = useRef<string>('Unknown');
  const deepFaceBoxesRef = useRef<{ x: number; y: number; w: number; h: number; identity: string; dni: string; age: number; gender: string; emotion: string; opacity: number }[]>([]);
  const lastDeepFaceTimeRef = useRef<number>(0);
  const localFaceDetectedRef = useRef<boolean>(false);

  useEffect(() => {
    analysisEngineRef.current = analysisEngine;
    if (analysisEngine === 'deepface') {
      setDeepFaceStatus('connecting');
      fetchLogs();
    } else {
      setLoggedInUser(null);
      setIdentity('Unknown');
      identityRef.current = 'Unknown';
      setIsFaceDetected(false);
      deepFaceBoxesRef.current = [];
      setDetectedFaces([]);
      setSelectedFace(null);
    }
  }, [analysisEngine]);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const faceCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionRef = useRef<any>(null);
  const playerRef = useRef<PCMPlayer | null>(null);
  
  const objectModelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const isPlayingRef = useRef(false);
  const lastPromptRef = useRef<string>("");
  const lastStateRef = useRef<string>("");
  const pendingStateRef = useRef<string | null>(null);
  const vibeTimeoutRef = useRef<any>(null);
  const lastStateUpdateTimeRef = useRef<number>(0);
  const detectLoopRef = useRef<number | null>(null);
  const smoothedBoxesRef = useRef<Map<string, SmoothedBox>>(new Map());
  const smoothedBlendshapesRef = useRef({ smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 });

  const playHoverSound = () => {
    try {
      initAudio();
      if (!hoverSynth || Tone.context.state !== 'running') return;
      
      const now = Tone.now();
      hoverSynth.triggerAttackRelease(800, 0.1, now);
      hoverSynth.frequency.exponentialRampToValueAtTime(1200, now + 0.1);
    } catch (e) {}
  };

  useEffect(() => {
    const handleInteraction = () => initAudio();
    window.addEventListener('click', handleInteraction, { once: true });
    window.addEventListener('touchstart', handleInteraction, { once: true });
    return () => {
      window.removeEventListener('click', handleInteraction);
      window.removeEventListener('touchstart', handleInteraction);
    };
  }, []);

  useEffect(() => {
    // Load TensorFlow, COCO-SSD, and MediaPipe FaceLandmarker
    const loadModels = async () => {
      try {
        await tf.ready();
        const cocoModel = await cocoSsd.load();
        objectModelRef.current = cocoModel;

        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1
        });
        faceLandmarkerRef.current = faceLandmarker;

        setIsModelLoaded(true);
        setStatus('Inactivo');
      } catch (err: any) {
        console.error("Failed to load models:", err);
        setStatus('Error al cargar modelos');
        setErrorMsg(err.message);
      }
    };
    
    loadModels();
    
    return () => {
      stopSession();
    };
  }, []);

  const runDetection = async () => {
    if (!isPlayingRef.current || !videoRef.current || !canvasRef.current || !objectModelRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (video.readyState >= 2 && ctx) {
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      try {
        const predictions = await objectModelRef.current.detect(video);
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const detectedClasses = new Set<string>();

        // --- Smoothing Logic ---
        const newSmoothedBoxes = new Map<string, SmoothedBox>();
        const unassignedPredictions = [...predictions];

        smoothedBoxesRef.current.forEach((box, id) => {
          let closestIdx = -1;
          let minDist = Infinity;
          unassignedPredictions.forEach((pred, idx) => {
            if (pred.class === box.class) {
              const [px, py, pw, ph] = pred.bbox;
              const dist = Math.hypot(px + pw/2 - (box.x + box.width/2), py + ph/2 - (box.y + box.height/2));
              if (dist < 150) {
                if (dist < minDist) {
                  minDist = dist;
                  closestIdx = idx;
                }
              }
            }
          });

          if (closestIdx !== -1) {
            const pred = unassignedPredictions[closestIdx];
            const [px, py, pw, ph] = pred.bbox;
            const lerp = 0.15; // Smoothing factor
            box.x += (px - box.x) * lerp;
            box.y += (py - box.y) * lerp;
            box.width += (pw - box.width) * lerp;
            box.height += (ph - box.height) * lerp;
            box.opacity = Math.min(1, box.opacity + 0.1);
            box.score = pred.score;
            
            // Target label position (top right of box)
            const targetLabelX = box.x + box.width + 20;
            const targetLabelY = box.y - 20;
            box.labelX += (targetLabelX - box.labelX) * lerp;
            box.labelY += (targetLabelY - box.labelY) * lerp;

            newSmoothedBoxes.set(id, box);
            unassignedPredictions.splice(closestIdx, 1);
            detectedClasses.add(box.class);
          } else {
            box.opacity -= 0.05; // Fade out
            if (box.opacity > 0) {
              newSmoothedBoxes.set(id, box);
              detectedClasses.add(box.class);
            }
          }
        });

        unassignedPredictions.forEach((pred) => {
          const id = Math.random().toString(36).substring(7);
          const [x, y, width, height] = pred.bbox;
          newSmoothedBoxes.set(id, {
            x, y, width, height, class: pred.class, score: pred.score, opacity: 0,
            labelX: x + width + 40, labelY: y - 40
          });
          detectedClasses.add(pred.class);
        });

        smoothedBoxesRef.current = newSmoothedBoxes;

        // --- Drawing Logic ---
        smoothedBoxesRef.current.forEach((box) => {
          const { x, y, width, height, opacity, labelX, labelY } = box;
          const text = `${box.class} (${Math.round(box.score * 100)}%)`;

          ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.8})`;
          ctx.lineWidth = 1;

          // Draw corners
          const cornerLength = Math.min(15, width / 4, height / 4);
          ctx.beginPath();
          ctx.moveTo(x, y + cornerLength);
          ctx.lineTo(x, y);
          ctx.lineTo(x + cornerLength, y);
          
          ctx.moveTo(x + width - cornerLength, y);
          ctx.lineTo(x + width, y);
          ctx.lineTo(x + width, y + cornerLength);
          
          ctx.moveTo(x + width, y + height - cornerLength);
          ctx.lineTo(x + width, y + height);
          ctx.lineTo(x + width - cornerLength, y + height);
          
          ctx.moveTo(x + cornerLength, y + height);
          ctx.lineTo(x, y + height);
          ctx.lineTo(x, y + height - cornerLength);
          ctx.stroke();

          // Crosshair center
          ctx.beginPath();
          ctx.moveTo(x + width / 2 - 5, y + height / 2);
          ctx.lineTo(x + width / 2 + 5, y + height / 2);
          ctx.moveTo(x + width / 2, y + height / 2 - 5);
          ctx.lineTo(x + width / 2, y + height / 2 + 5);
          ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.4})`;
          ctx.stroke();

          // Line to label
          ctx.beginPath();
          ctx.moveTo(x + width, y);
          ctx.lineTo(labelX, labelY + 16);
          ctx.strokeStyle = `rgba(255, 255, 255, ${opacity * 0.5})`;
          ctx.setLineDash([2, 2]);
          ctx.stroke();
          ctx.setLineDash([]);

          // Minimalist Label
          ctx.font = '400 10px "JetBrains Mono", monospace';
          const textWidth = ctx.measureText(text).width;
          ctx.fillStyle = `rgba(255, 255, 255, ${opacity * 0.2})`;
          ctx.fillRect(labelX, labelY, textWidth + 8, 16);
          ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
          ctx.fillText(text.toUpperCase(), labelX + 4, labelY + 11);
        });

        const classesArray = Array.from(detectedClasses).sort();
        
        let currentEmotion = "neutral";
        let currentBlendshapes = { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 };
        let localFaceDetected = false;

        if (faceLandmarkerRef.current) {
          const faceResult = faceLandmarkerRef.current.detectForVideo(video, performance.now());
          
          if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
            localFaceDetected = true;

            // Biometric Quality Assessment & Head Pose Estimation
            const landmarks = faceResult.faceLandmarks[0];
            const leftEye = landmarks[33];
            const rightEye = landmarks[263];
            const noseTip = landmarks[4];
            const leftEdge = landmarks[234];
            const rightEdge = landmarks[454];
            const forehead = landmarks[10];
            const chin = landmarks[152];
            
            const rollRad = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);
            const roll = rollRad * (180 / Math.PI);
            
            const leftDist = noseTip.x - leftEdge.x;
            const rightDist = rightEdge.x - noseTip.x;
            const yawRatio = leftDist / (leftDist + rightDist);
            const yaw = (yawRatio - 0.5) * 100;
            
            const upperDist = noseTip.y - forehead.y;
            const lowerDist = chin.y - noseTip.y;
            const pitchRatio = upperDist / (upperDist + lowerDist);
            const pitch = (pitchRatio - 0.58) * 100;

            // Calculate face box dimensions for brightness checks
            let fMinX = 1, fMaxX = 0, fMinY = 1, fMaxY = 0;
            for (const pt of landmarks) {
              if (pt.x < fMinX) fMinX = pt.x;
              if (pt.x > fMaxX) fMaxX = pt.x;
              if (pt.y < fMinY) fMinY = pt.y;
              if (pt.y > fMaxY) fMaxY = pt.y;
            }
            const fx = fMinX * video.videoWidth;
            const fy = fMinY * video.videoHeight;
            const fw = (fMaxX - fMinX) * video.videoWidth;
            const fh = (fMaxY - fMinY) * video.videoHeight;

            // Sample brightness from the canvas inside the face area
            let avgBrightness = 120;
            try {
              const sampleW = Math.floor(fw / 2);
              const sampleH = Math.floor(fh / 2);
              const sampleX = Math.floor(fx + fw / 4);
              const sampleY = Math.floor(fy + fh / 4);
              
              if (sampleW > 5 && sampleH > 5 && sampleX >= 0 && sampleY >= 0 && sampleX + sampleW <= canvas.width && sampleY + sampleH <= canvas.height) {
                const imgData = ctx.getImageData(sampleX, sampleY, sampleW, sampleH);
                const pData = imgData.data;
                let brightnessSum = 0;
                for (let i = 0; i < pData.length; i += 4) {
                  brightnessSum += (pData[i] + pData[i+1] + pData[i+2]) / 3;
                }
                avgBrightness = brightnessSum / (pData.length / 4);
              }
            } catch (e) {}

            // Detect expression (using faceResult categories)
            let smileScore = 0;
            let frownScore = 0;
            let blinkScore = 0;
            if (faceResult.faceBlendshapes && faceResult.faceBlendshapes.length > 0) {
              const blendshapes = faceResult.faceBlendshapes[0].categories;
              smileScore = ((blendshapes.find(b => b.categoryName === 'mouthSmileLeft')?.score || 0) + (blendshapes.find(b => b.categoryName === 'mouthSmileRight')?.score || 0)) / 2;
              frownScore = Math.min(1, ((blendshapes.find(b => b.categoryName === 'mouthFrownLeft')?.score || 0) + (blendshapes.find(b => b.categoryName === 'mouthFrownRight')?.score || 0) + (blendshapes.find(b => b.categoryName === 'mouthRollLower')?.score || 0)) * 5);
              blinkScore = ((blendshapes.find(b => b.categoryName === 'eyeBlinkLeft')?.score || 0) + (blendshapes.find(b => b.categoryName === 'eyeBlinkRight')?.score || 0)) / 2;
            }

            const isNeutral = smileScore < 0.18 && frownScore < 0.18;
            let qualityWarning = '';
            if (avgBrightness < 60) {
              qualityWarning = "ILUMINACIÓN INSUFICIENTE (MUY OSCURO)";
            } else if (avgBrightness > 215) {
              qualityWarning = "SOBREEXPOSICIÓN (MUCHA LUZ)";
            } else if (!isNeutral) {
              qualityWarning = "EXPRESIÓN NEUTRAL REQUERIDA (NO SONRÍAS / FRUNZAS)";
            }

            // Sync with state periodically (throttled)
            if (Math.random() < 0.15) {
              setHudPose({ yaw, roll, pitch });
              setHudQuality({ brightness: avgBrightness, isNeutral, warning: qualityWarning });
            }

            // Interactive steps handler
            const currentStep = enrollmentStepRef.current;
            if (currentStep !== 'idle') {
              if (currentStep === 'liveness') {
                if (blinkScore > 0.65) {
                  enrollmentStepRef.current = 'frontal';
                  setEnrollmentStep('frontal');
                  setRegistrationProgress(20);
                  setRegistrationStatusText("ÁNGULO 1/3: MIRA AL CENTRO DEL ÓVALO CON EXPRESIÓN NEUTRAL...");
                }
              } else if (currentStep === 'frontal') {
                if (qualityWarning === '') {
                  if (Math.abs(yaw) < 4 && Math.abs(roll) < 5 && Math.abs(pitch) < 5) {
                    const img = captureCurrentFaceCrop({ x: fx, y: fy, w: fw, h: fh });
                    if (img) {
                      enrollmentPhotosRef.current.push(img);
                      enrollmentStepRef.current = 'left';
                      setEnrollmentStep('left');
                      setRegistrationProgress(50);
                      setRegistrationStatusText("ÁNGULO 2/3: GIRA LENTAMENTE LA CABEZA A LA IZQUIERDA...");
                    }
                  }
                }
              } else if (currentStep === 'left') {
                if (yaw < -8) {
                  const img = captureCurrentFaceCrop({ x: fx, y: fy, w: fw, h: fh });
                  if (img) {
                    enrollmentPhotosRef.current.push(img);
                    enrollmentStepRef.current = 'right';
                    setEnrollmentStep('right');
                    setRegistrationProgress(80);
                    setRegistrationStatusText("ÁNGULO 3/3: GIRA LENTAMENTE LA CABEZA A LA DERECHA...");
                  }
                }
              } else if (currentStep === 'right') {
                if (yaw > 8) {
                  const img = captureCurrentFaceCrop({ x: fx, y: fy, w: fw, h: fh });
                  if (img) {
                    enrollmentPhotosRef.current.push(img);
                    enrollmentStepRef.current = 'submitting';
                    setEnrollmentStep('submitting');
                    setRegistrationProgress(95);
                    setRegistrationStatusText("PROCESANDO Y REGISTRANDO PERFIL BIOMÉTRICO...");
                    submitEnrollment(firstName.trim(), lastName.trim(), dni.trim());
                  }
                }
              }
            }
            
            // Draw Face Mesh Point Cloud on secondary canvas
            if (faceCanvasRef.current) {
              const fCanvas = faceCanvasRef.current;
              const fCtx = fCanvas.getContext('2d');
              if (fCtx) {
                fCtx.clearRect(0, 0, fCanvas.width, fCanvas.height);
                
                const time = performance.now() / 1500; // 1.5 seconds per cycle
                
                // Find bounding box of face to center it
                let minX = video.videoWidth, maxX = 0, minY = video.videoHeight, maxY = 0;
                for (const pt of faceResult.faceLandmarks[0]) {
                  const px = pt.x * video.videoWidth;
                  const py = pt.y * video.videoHeight;
                  if (px < minX) minX = px;
                  if (px > maxX) maxX = px;
                  if (py < minY) minY = py;
                  if (py > maxY) maxY = py;
                }
                const faceWidth = maxX - minX;
                const faceHeight = maxY - minY;
                const centerX = minX + faceWidth / 2;
                const centerY = minY + faceHeight / 2;
                
                const scanY = minY + ((Math.sin(time) + 1) / 2) * faceHeight;
                const scale = Math.min(fCanvas.width / faceWidth, fCanvas.height / faceHeight) * 0.8;

                for (const pt of faceResult.faceLandmarks[0]) {
                  const px = pt.x * video.videoWidth;
                  const py = pt.y * video.videoHeight;
                  
                  // Distance from scan line (in pixels)
                  const dist = Math.abs(py - scanY) / faceHeight;
                  // Opacity: high near scan line, low elsewhere
                  const opacity = Math.max(0.15, 1.0 - dist * 4); 
                  
                  fCtx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
                  fCtx.beginPath();
                  
                  // Map to canvas
                  const drawX = fCanvas.width/2 + (px - centerX) * scale;
                  const drawY = fCanvas.height/2 + (py - centerY) * scale;
                  
                  fCtx.arc(drawX, drawY, 1.5, 0, 2 * Math.PI);
                  fCtx.fill();
                }
              }
            }

            if (faceResult.faceBlendshapes && faceResult.faceBlendshapes.length > 0) {
              const blendshapes = faceResult.faceBlendshapes[0].categories;
              const getScore = (name: string) => blendshapes.find(b => b.categoryName === name)?.score || 0;
              
              // Physical facial features for sliders
              currentBlendshapes.smile = (getScore('mouthSmileLeft') + getScore('mouthSmileRight')) / 2;
              currentBlendshapes.frown = Math.min(1, (getScore('mouthFrownLeft') + getScore('mouthFrownRight') + getScore('mouthRollLower')) * 5);
              currentBlendshapes.mouthOpen = getScore('jawOpen');
              currentBlendshapes.browRaise = (getScore('browInnerUp') + getScore('browOuterUpLeft') + getScore('browOuterUpRight')) / 3;
              currentBlendshapes.eyeBlink = (getScore('eyeBlinkLeft') + getScore('eyeBlinkRight')) / 2;
              currentBlendshapes.pucker = getScore('mouthPucker');
              
              // High-level emotions for the overall state
              const surpriseScore = (getScore('jawOpen') + getScore('browInnerUp')) / 2;
              const angerScore = (getScore('browDownLeft') + getScore('browDownRight') + getScore('mouthPressLeft')) / 3;
              const fearScore = ((getScore('jawOpen') + getScore('browInnerUp') + getScore('mouthStretchLeft') + getScore('mouthStretchRight')) / 4) * 0.6;
              const disgustScore = Math.min(1, (getScore('noseSneerLeft') + getScore('noseSneerRight') + getScore('mouthUpperUpLeft') + getScore('mouthUpperUpRight')) * 4);
              
              const emotions = [
                { name: 'happy', score: currentBlendshapes.smile },
                { name: 'sadness', score: currentBlendshapes.frown },
                { name: 'surprised', score: surpriseScore },
                { name: 'angry', score: angerScore },
                { name: 'fear', score: fearScore },
                { name: 'disgust', score: disgustScore }
              ];
              
              const maxEmotion = emotions.reduce((max, e) => e.score > max.score ? e : max, emotions[0]);
              if (maxEmotion.score > 0.2) {
                currentEmotion = maxEmotion.name;
              } else {
                currentEmotion = "neutral";
              }
            }

            // Draw arrows and blurred line on main canvas
            // landmarks is already defined above as faceResult.faceLandmarks[0]
            
            // Draw blurred line scanning over the face every 40 seconds
            const scanTime = performance.now() / 40000; // 40 seconds
            const scanPhase = scanTime % 1; // 0 to 1
            
            let minX = 1, maxX = 0, minY = 1, maxY = 0;
            for (const pt of landmarks) {
              if (pt.x < minX) minX = pt.x;
              if (pt.x > maxX) maxX = pt.x;
              if (pt.y < minY) minY = pt.y;
              if (pt.y > maxY) maxY = pt.y;
            }
            
            // Oscillate the scan line up and down
            const scanProgress = (Math.sin(scanPhase * Math.PI * 2) + 1) / 2; // 0 to 1 to 0
            const scanY = minY + scanProgress * (maxY - minY);
            
            // Opacity: 0 at top/bottom, 0.3 in the middle
            const lineOpacity = Math.sin(scanProgress * Math.PI) * 0.3;
            
            ctx.save();
            
            // Clip to face oval
            const faceOvalIndices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
            ctx.beginPath();
            for (let i = 0; i < faceOvalIndices.length; i++) {
              const pt = landmarks[faceOvalIndices[i]];
              if (i === 0) ctx.moveTo(pt.x * canvas.width, pt.y * canvas.height);
              else ctx.lineTo(pt.x * canvas.width, pt.y * canvas.height);
            }
            ctx.closePath();
            ctx.clip();

            if (lineOpacity > 0.01) {
              // 1. Draw the actual face landmarks illuminated by the scanner
              ctx.fillStyle = '#ffffff'; // White glow
              ctx.shadowColor = '#ffffff';
              ctx.shadowBlur = 10;
              
              for (const pt of landmarks) {
                // Calculate vertical distance from the scan line
                const dist = Math.abs(pt.y - scanY);
                const threshold = 0.04; // How thick the illuminated band is
                
                if (dist < threshold) {
                  // Opacity falls off as points get further from the scan line
                  const ptOpacity = (1 - (dist / threshold)) * lineOpacity * 2;
                  ctx.globalAlpha = Math.min(ptOpacity, 1);
                  
                  ctx.beginPath();
                  ctx.arc(pt.x * canvas.width, pt.y * canvas.height, 1.5, 0, Math.PI * 2);
                  ctx.fill();
                }
              }
              ctx.globalAlpha = 1.0;
            }
            
            ctx.restore();

            // Draw stylized feature highlights based on emotion
            const drawFeatureHighlight = (x: number, y: number, label: string, intensity: number) => {
              if (isNaN(intensity) || intensity < 0.05) return;
              ctx.save();
              ctx.translate(x * canvas.width, y * canvas.height);
              
              const size = 5 + intensity * 15;
              
              ctx.strokeStyle = `rgba(255, 255, 255, ${intensity * 0.8})`;
              ctx.lineWidth = 1.5;
              
              // Draw brackets [ ]
              ctx.beginPath();
              ctx.moveTo(-size, -size/2);
              ctx.lineTo(-size, -size);
              ctx.lineTo(-size/2, -size);
              
              ctx.moveTo(size, -size/2);
              ctx.lineTo(size, -size);
              ctx.lineTo(size/2, -size);
              
              ctx.moveTo(-size, size/2);
              ctx.lineTo(-size, size);
              ctx.lineTo(-size/2, size);
              
              ctx.moveTo(size, size/2);
              ctx.lineTo(size, size);
              ctx.lineTo(size/2, size);
              ctx.stroke();
              
              // Center dot
              ctx.fillStyle = `rgba(255, 255, 255, ${intensity})`;
              ctx.beginPath();
              ctx.arc(0, 0, 2, 0, Math.PI * 2);
              ctx.fill();
              
              // Label
              ctx.fillStyle = `rgba(255, 255, 255, ${intensity * 0.9})`;
              ctx.font = '10px monospace';
              ctx.fillText(label, size + 5, 3);
              
              ctx.restore();
            };

            const smoothed = smoothedBlendshapesRef.current;
            drawFeatureHighlight(landmarks[61].x, landmarks[61].y, 'SONRISA_I', smoothed.smile);
            drawFeatureHighlight(landmarks[291].x, landmarks[291].y, 'SONRISA_D', smoothed.smile);
            
            drawFeatureHighlight(landmarks[61].x, landmarks[61].y, 'CEÑO_I', smoothed.frown);
            drawFeatureHighlight(landmarks[291].x, landmarks[291].y, 'CEÑO_D', smoothed.frown);
            
            drawFeatureHighlight(landmarks[52].x, landmarks[52].y, 'CEJA_I', smoothed.browRaise);
            drawFeatureHighlight(landmarks[282].x, landmarks[282].y, 'CEJA_D', smoothed.browRaise);
            
            drawFeatureHighlight(landmarks[152].x, landmarks[152].y, 'BOCA_ABIERTA', smoothed.mouthOpen);
            
            drawFeatureHighlight(landmarks[13].x, landmarks[13].y, 'CEÑO_BOCA', smoothed.pucker);
            
            drawFeatureHighlight(landmarks[159].x, landmarks[159].y, 'PARPADEO_I', smoothed.eyeBlink);
            drawFeatureHighlight(landmarks[386].x, landmarks[386].y, 'PARPADEO_D', smoothed.eyeBlink);
          }
        }

        // Set the ref so async fetch callbacks can read it
        localFaceDetectedRef.current = localFaceDetected;

        if (analysisEngineRef.current === 'mediapipe') {
          // In MediaPipe mode, we already finished tracking and rendering!
          if (!localFaceDetected && faceCanvasRef.current) {
            const fCanvas = faceCanvasRef.current;
            const fCtx = fCanvas.getContext('2d');
            if (fCtx) {
              fCtx.clearRect(0, 0, fCanvas.width, fCanvas.height);
            }
          }
        } else if (analysisEngineRef.current === 'deepface') {
          // If we didn't detect a face locally, use the server-estimated values (which were set in the background fetch callback)
          if (!localFaceDetected) {
            currentEmotion = deepFaceEmotionRef.current || "neutral";
            currentBlendshapes = { ...smoothedBlendshapesRef.current };
          }

          // Trigger DeepFace processing asynchronously every 750ms
          const nowMs = performance.now();
          if (nowMs - lastDeepFaceTimeRef.current > 750) {
            lastDeepFaceTimeRef.current = nowMs;
            
            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = video.videoWidth || 640;
            captureCanvas.height = video.videoHeight || 480;
            const cCtx = captureCanvas.getContext('2d');
            if (cCtx) {
              cCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
              const base64Image = captureCanvas.toDataURL('image/jpeg', 0.8);
              
              fetch('http://localhost:5000/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image })
              })
              .then(res => res.json())
              .then(data => {
                setDeepFaceStatus('connected');
                if (data.results && data.results.length > 0) {
                  setIsFaceDetected(true);
                  
                  // Filter out active users (recognized ones)
                  const detectedUsers = data.results
                    .filter((face: any) => face.identity !== 'Unknown')
                    .map((face: any) => ({ name: face.identity, dni: face.dni }));
                  setActiveUsers(detectedUsers);
                  
                  setDetectedFaces(data.results.map((face: any) => ({
                    box: face.box,
                    identity: face.identity,
                    dni: face.dni,
                    age: face.age,
                    gender: face.dominant_gender,
                    emotion: face.dominant_emotion
                  })));
                  
                  // Set primary log in user as the first recognized user, if any
                  if (detectedUsers.length > 0) {
                    setLoggedInUser(detectedUsers[0].name);
                    setLoggedInDni(detectedUsers[0].dni);
                    setIdentity(detectedUsers[0].name);
                    identityRef.current = detectedUsers[0].name;
                  } else {
                    const firstFace = data.results[0];
                    setIdentity(firstFace.identity);
                    identityRef.current = firstFace.identity;
                  }

                  const firstFace = data.results[0];
                  deepFaceEmotionRef.current = firstFace.dominant_emotion;
                  setDemographics({
                    age: firstFace.age,
                    gender: firstFace.dominant_gender
                  });

                  // Update normalized values for sliders ONLY if local tracking is not active
                  if (!localFaceDetectedRef.current) {
                    smoothedBlendshapesRef.current.smile = (firstFace.emotion.happy || 0) / 100;
                    smoothedBlendshapesRef.current.frown = (firstFace.emotion.sad || 0) / 100;
                    smoothedBlendshapesRef.current.mouthOpen = (firstFace.emotion.surprise || 0) / 100;
                    smoothedBlendshapesRef.current.browRaise = ((firstFace.emotion.surprise || 0) + (firstFace.emotion.fear || 0)) / 200;
                    smoothedBlendshapesRef.current.pucker = (firstFace.emotion.disgust || 0) / 100;
                    smoothedBlendshapesRef.current.eyeBlink = 0.1;
                  }

                  // Update deepFaceBoxesRef with all detected faces
                  deepFaceBoxesRef.current = data.results.map((face: any) => ({
                    x: face.box.x,
                    y: face.box.y,
                    w: face.box.w,
                    h: face.box.h,
                    identity: face.identity,
                    dni: face.dni,
                    age: face.age,
                    gender: face.dominant_gender,
                    emotion: face.dominant_emotion,
                    opacity: 1.0
                  }));
                } else {
                  setIsFaceDetected(false);
                  setIdentity('Unknown');
                  identityRef.current = 'Unknown';
                  setLoggedInDni(null);
                  deepFaceEmotionRef.current = 'neutral';
                  setDemographics(null);
                  setDetectedFaces([]);
                  
                  // Decrease opacity for all existing boxes
                  deepFaceBoxesRef.current = deepFaceBoxesRef.current
                    .map(box => ({ ...box, opacity: box.opacity - 0.15 }))
                    .filter(box => box.opacity > 0);
                }
                fetchLogs();
              })
              .catch(err => {
                console.error("DeepFace backend error:", err);
                setDeepFaceStatus('offline');
                setIsFaceDetected(false);
                setIdentity('Unknown');
                identityRef.current = 'Unknown';
                setDetectedFaces([]);
              });
            }
          }

          // Draw stylized mesh point cloud animation on the secondary canvas only if no local face is detected
          if (faceCanvasRef.current && !localFaceDetected) {
            const fCanvas = faceCanvasRef.current;
            const fCtx = fCanvas.getContext('2d');
            if (fCtx) {
              fCtx.clearRect(0, 0, fCanvas.width, fCanvas.height);
              
              const activeBox = deepFaceBoxesRef.current[0];
              if (activeBox && activeBox.opacity > 0) {
                const time = performance.now() / 1500;
                const scanProgress = (Math.sin(time * Math.PI * 2) + 1) / 2;
                
                fCtx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
                fCtx.lineWidth = 1;
                
                // Draw radar target rings
                fCtx.beginPath();
                fCtx.arc(fCanvas.width/2, fCanvas.height/2, 55, 0, Math.PI * 2);
                fCtx.stroke();
                
                fCtx.beginPath();
                fCtx.arc(fCanvas.width/2, fCanvas.height/2, 30, 0, Math.PI * 2);
                fCtx.stroke();
                
                fCtx.beginPath();
                fCtx.moveTo(fCanvas.width/2 - 70, fCanvas.height/2);
                fCtx.lineTo(fCanvas.width/2 + 70, fCanvas.height/2);
                fCtx.moveTo(fCanvas.width/2, fCanvas.height/2 - 70);
                fCtx.lineTo(fCanvas.width/2, fCanvas.height/2 + 70);
                fCtx.stroke();
                
                // Draw biometric constellation
                const points = getConstellationPoints(identityRef.current, performance.now());
                const scanLineY = -60 + scanProgress * 120; // local offset from center
                
                // Draw connections
                fCtx.strokeStyle = `rgba(255, 255, 255, ${0.12 * activeBox.opacity})`;
                fCtx.lineWidth = 0.5;
                for (let i = 0; i < points.length; i++) {
                  for (let j = i + 1; j < points.length; j++) {
                    const dx = points[i].x - points[j].x;
                    const dy = points[i].y - points[j].y;
                    const dist = Math.hypot(dx, dy);
                    if (dist < 18) {
                      fCtx.beginPath();
                      fCtx.moveTo(fCanvas.width/2 + points[i].x, fCanvas.height/2 + points[i].y);
                      fCtx.lineTo(fCanvas.width/2 + points[j].x, fCanvas.height/2 + points[j].y);
                      fCtx.stroke();
                    }
                  }
                }
                
                // Draw points illuminated by scanline
                points.forEach((pt) => {
                  const distToScanline = Math.abs(pt.y - scanLineY);
                  // Highlight points closer to scanline
                  const brightness = Math.max(0.2, 1.0 - distToScanline / 25);
                  
                  fCtx.fillStyle = `rgba(255, 255, 255, ${brightness * activeBox.opacity})`;
                  fCtx.beginPath();
                  fCtx.arc(fCanvas.width/2 + pt.x, fCanvas.height/2 + pt.y, pt.size, 0, Math.PI * 2);
                  fCtx.fill();
                });
              }
            }
          }

          // Draw Bounding Boxes for DeepFace on main canvas
          deepFaceBoxesRef.current.forEach((box) => {
            if (box.opacity <= 0) return;
            const { x, y, w, h, identity, dni, opacity } = box;
            
            const isUnknown = identity === 'Unknown';
            
            // Check if this box is the selected one
            const isSelected = selectedFace && 
              Math.abs(selectedFace.x - x) < 20 && 
              Math.abs(selectedFace.y - y) < 20;

            // Glowing green for authorized (recognized), warning yellow for unrecognized, bright blue for selected
            let boxColor = isUnknown ? `rgba(234, 179, 8, ${opacity})` : `rgba(74, 222, 128, ${opacity})`;
            let bgBoxColor = isUnknown ? `rgba(234, 179, 8, ${opacity * 0.15})` : `rgba(74, 222, 128, ${opacity * 0.15})`;
            
            if (isSelected) {
              boxColor = `rgba(59, 130, 246, ${opacity})`; // Blue
              bgBoxColor = `rgba(59, 130, 246, ${opacity * 0.3})`;
            }

            ctx.strokeStyle = boxColor;
            ctx.lineWidth = isSelected ? 2.5 : 1.5;
            
            const cornerLength = Math.min(15, w / 4, h / 4);
            ctx.beginPath();
            ctx.moveTo(x, y + cornerLength);
            ctx.lineTo(x, y);
            ctx.lineTo(x + cornerLength, y);
            
            ctx.moveTo(x + w - cornerLength, y);
            ctx.lineTo(x + w, y);
            ctx.lineTo(x + w, y + cornerLength);
            
            ctx.moveTo(x + w, y + h - cornerLength);
            ctx.lineTo(x + w, y + h);
            ctx.lineTo(x + w - cornerLength, y + h);
            
            ctx.moveTo(x + cornerLength, y + h);
            ctx.lineTo(x, y + h);
            ctx.lineTo(x, y + h - cornerLength);
            ctx.stroke();

            // If selected, draw a dashed box connecting the corners as well
            if (isSelected) {
              ctx.save();
              ctx.strokeStyle = `rgba(59, 130, 246, ${opacity * 0.5})`;
              ctx.setLineDash([4, 4]);
              ctx.strokeRect(x, y, w, h);
              ctx.restore();
            }
            
            // Draw name and DNI above the bounding box
            ctx.font = '700 9px "JetBrains Mono", monospace';
            let labelText = isUnknown ? "SUJETO_NO_REGISTRADO" : identity.toUpperCase();
            let subLabelText = isUnknown ? "ADVERTENCIA: NO AUTORIZADO" : `DNI: ${dni || 'S/D'}`;
            
            if (isSelected) {
              labelText = "REGISTRANDO SUJETO...";
              subLabelText = "SELECCIONADO EN PANEL";
            }
            
            const labelWidth = Math.max(ctx.measureText(labelText).width, ctx.measureText(subLabelText).width) + 12;
            const labelHeight = 28;
            
            // Position label above box, or inside at the top if there is no space above
            const labelY = y - labelHeight - 6 > 0 ? y - labelHeight - 6 : y + 6;
            const labelX = x;
            
            ctx.fillStyle = bgBoxColor;
            ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
            
            ctx.strokeStyle = boxColor;
            ctx.lineWidth = 1;
            ctx.strokeRect(labelX, labelY, labelWidth, labelHeight);
            
            ctx.fillStyle = boxColor;
            ctx.fillText(labelText, labelX + 6, labelY + 11);
            
            ctx.font = '400 8px "JetBrains Mono", monospace';
            ctx.fillStyle = (isUnknown && !isSelected) ? `rgba(234, 179, 8, ${opacity * 0.8})` : isSelected ? `rgba(59, 130, 246, ${opacity * 0.8})` : `rgba(255, 255, 255, ${opacity * 0.8})`;
            ctx.fillText(subLabelText, labelX + 6, labelY + 22);
          });

          // Draw guide box/oval for face alignment during registration/scanning
          // Disappears once a face is selected, but reappears during active enrollment steps
          if (!loggedInUser && (selectedFace === null || enrollmentStepRef.current !== 'idle')) {
            const boxWidth = 240;
            const boxHeight = 290;
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;
            const x = cx - boxWidth / 2;
            const y = cy - boxHeight / 2;
            
            const time = performance.now() / 1000;
            // Pulsing animation for breathing effect
            const pulse = 0.4 + Math.sin(time * 3.5) * 0.15;
            
            // Draw a dashed ellipse in the center
            ctx.save();
            ctx.strokeStyle = `rgba(234, 179, 8, ${pulse})`; // Glowing amber/yellow
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 6]);
            ctx.beginPath();
            ctx.ellipse(cx, cy, boxWidth / 2, boxHeight / 2, 0, 0, 2 * Math.PI);
            ctx.stroke();
            ctx.restore();
            
            // Draw stylized retro corner brackets
            const bracketLength = 20;
            const padding = 15;
            const bx = x - padding;
            const by = y - padding;
            const bw = boxWidth + padding * 2;
            const bh = boxHeight + padding * 2;
            
            ctx.strokeStyle = `rgba(234, 179, 8, 0.45)`;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            // Top Left
            ctx.moveTo(bx, by + bracketLength); ctx.lineTo(bx, by); ctx.lineTo(bx + bracketLength, by);
            // Top Right
            ctx.moveTo(bx + bw - bracketLength, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + bracketLength);
            // Bottom Left
            ctx.moveTo(bx, by + bh - bracketLength); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + bracketLength, by + bh);
            // Bottom Right
            ctx.moveTo(bx + bw - bracketLength, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - bracketLength);
            ctx.stroke();
            
            // Guide text labels
            ctx.font = '700 9px "JetBrains Mono", monospace';
            ctx.fillStyle = `rgba(234, 179, 8, ${pulse * 1.3})`;
            const labelText1 = "ALINEAR ROSTRO EN EL ÓVALO";
            const labelText2 = "SISTEMA DE REGISTRO BIOMÉTRICO";
            const w1 = ctx.measureText(labelText1).width;
            const w2 = ctx.measureText(labelText2).width;
            
            ctx.fillText(labelText1, cx - w1 / 2, by - 8);
            
            ctx.font = '500 7px "JetBrains Mono", monospace';
            ctx.fillStyle = `rgba(255, 255, 255, 0.5)`;
            ctx.fillText(labelText2, cx - w2 / 2, by + bh + 14);
          }
        }

        // Throttle React state updates for the console UI to ~10fps
        const now = performance.now();
        if (now - lastStateUpdateTimeRef.current > 100) {
          const smoothingFactor = 0.15;
          const smoothed = smoothedBlendshapesRef.current;
          smoothed.smile += (currentBlendshapes.smile - smoothed.smile) * smoothingFactor;
          smoothed.frown += (currentBlendshapes.frown - smoothed.frown) * smoothingFactor;
          smoothed.mouthOpen += (currentBlendshapes.mouthOpen - smoothed.mouthOpen) * smoothingFactor;
          smoothed.browRaise += (currentBlendshapes.browRaise - smoothed.browRaise) * smoothingFactor;
          smoothed.eyeBlink += (currentBlendshapes.eyeBlink - smoothed.eyeBlink) * smoothingFactor;
          smoothed.pucker += (currentBlendshapes.pucker - smoothed.pucker) * smoothingFactor;

          setConsoleState({
            emotion: currentEmotion,
            objects: classesArray,
            blendshapes: { ...smoothed }
          });
          lastStateUpdateTimeRef.current = now;
        }

        const stateString = `${classesArray.join(',')}|${currentEmotion}|${identityRef.current}`;
        
        if (stateString !== pendingStateRef.current) {
          pendingStateRef.current = stateString;
          
          if (vibeTimeoutRef.current) {
            clearTimeout(vibeTimeoutRef.current);
          }
          
          // Debounce prompt changes by 3 seconds to avoid flickering
          vibeTimeoutRef.current = setTimeout(async () => {
            if (stateString !== lastStateRef.current) {
              lastStateRef.current = stateString;
              
              const newVibe = await getVibeFromGemini(classesArray, currentEmotion, identityRef.current);
              lastPromptRef.current = newVibe;
              setCurrentPrompt(newVibe);
              
              if (sessionRef.current) {
                sessionRef.current.setWeightedPrompts({
                  weightedPrompts: [{ text: newVibe, weight: 1.0 }]
                }).catch(console.error);
              }
            }
          }, 3000);
        }
      } catch (err) {
        console.error("Detection error:", err);
      }
    }
    
    if (isPlayingRef.current) {
      detectLoopRef.current = requestAnimationFrame(runDetection);
    }
  };

  const startSession = async () => {
    if (!isModelLoaded) return;
    
    try {
      setErrorMsg(null);
      setStatus('Starting camera...');
      
      let stream = streamRef.current;
      if (!stream) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
              width: { ideal: 1280 }, 
              height: { ideal: 720 },
              facingMode: 'user'
            } 
          });
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play().catch(e => console.error("Video play error:", e));
          }
          setIsCameraActive(true);
        } catch (camErr: any) {
          console.error("Camera error:", camErr);
          setStatus('Camera Error');
          setErrorMsg('Camera access denied. Please allow camera access in your browser settings, then refresh the browser page.');
          return;
        }
      }

      // Start detection immediately so it runs even if Lyria fails
      if (!isPlayingRef.current) {
        isPlayingRef.current = true;
        detectLoopRef.current = requestAnimationFrame(runDetection);
      }

      setStatus('Connecting to Lyria API...');
      playerRef.current = new PCMPlayer(48000);

      let timeoutId: any;

      let sessionPromise;
      try {
        const ai = new GoogleGenAI({ 
          apiKey: process.env.API_KEY || process.env.GEMINI_API_KEY || 'dummy-key',
          apiVersion: 'v1alpha'
        });

        sessionPromise = ai.live.music.connect({
          model: "lyria-realtime-exp",
          callbacks: {
            onmessage: (message: any) => {
              if (message.setupComplete) {
                console.log("Lyria setup complete");
              }
              const audioChunk = message.audioChunk;
              if (audioChunk?.data && playerRef.current) {
                playerRef.current.playChunk(audioChunk.data);
              }
            },
            onclose: () => {
              clearTimeout(timeoutId);
              if (isPlayingRef.current) {
                setInfoMsg('Lyria connection closed.');
                stopSession(false);
              } else {
                stopSession(false);
              }
            },
            onerror: (err: any) => {
              clearTimeout(timeoutId);
              console.error("Lyria API Error:", err);
              setErrorMsg(err.message || 'Connection error with Lyria API.');
              setInfoMsg(null);
              stopSession(false);
            }
          }
        });
      } catch (err: any) {
        console.warn("Failed to initialize Lyria API:", err);
        clearTimeout(timeoutId);
        setErrorMsg('API key missing or invalid.');
        setInfoMsg(null);
        stopSession(false);
        return;
      }
      
      timeoutId = setTimeout(() => {
        setErrorMsg('Lyria API took too long to respond.');
        setInfoMsg(null);
        stopSession(false);
      }, 30000);

      sessionPromise.then(async session => {
        clearTimeout(timeoutId);
        sessionRef.current = session;
        setStatus('Connected & Playing');
        setIsPlaying(true);
        
        const initialPrompt = "minimalist ambient drone, quiet";
        setCurrentPrompt(initialPrompt);
        lastPromptRef.current = initialPrompt;
        
        try {
          await session.setMusicGenerationConfig({
            musicGenerationConfig: { bpm: 120, temperature: 1.0 }
          });
          await session.setWeightedPrompts({
            weightedPrompts: [{ text: initialPrompt, weight: 1.0 }]
          });
          session.play();
        } catch (e) {
          console.error("Error setting up session:", e);
        }
      }).catch(err => {
        clearTimeout(timeoutId);
        console.error("API Error:", err);
        if (err.message?.includes('403') || err.message?.includes('Permission denied') || err.status === 403) {
          setErrorMsg('Lyria RealTime is experimental and requires allowlisting.');
        } else {
          setErrorMsg(err.message || 'An unknown API error occurred.');
        }
        setInfoMsg(null);
        stopSession(false);
      });

    } catch (err: any) {
      console.error("Setup Error:", err);
      setStatus('Failed to connect');
      setErrorMsg(err.message || 'An unknown error occurred during setup.');
      setInfoMsg(null);
      stopSession(false);
    }
  };

  const stopSession = (closeCamera: boolean = true) => {
    setIsPlaying(false);
    if (vibeTimeoutRef.current) {
      clearTimeout(vibeTimeoutRef.current);
      vibeTimeoutRef.current = null;
    }
    pendingStateRef.current = null;
    
    if (status === 'Conectado y Reproduciendo' || status === 'Conectando con la API de Lyria...' || status.includes('Sintetizador Local') || status.includes('Local Synth')) {
      setStatus('Inactivo');
    }
    
    if (playerRef.current) {
      playerRef.current.stop();
      playerRef.current = null;
    }
    if (sessionRef.current) {
      try { sessionRef.current.conn.close(); } catch (e) {}
      sessionRef.current = null;
    }
    
    setConsoleState({
      emotion: 'neutral',
      objects: [],
      blendshapes: { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 }
    });
    smoothedBlendshapesRef.current = { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 };
    smoothedBoxesRef.current.clear();
    deepFaceBoxesRef.current = [];
    setDetectedFaces([]);
    setSelectedFace(null);
    
    setInfoMsg(null);

    if (closeCamera) {
      isPlayingRef.current = false;
      setCurrentPrompt('Waiting for camera...');
      
      if (detectLoopRef.current) {
        cancelAnimationFrame(detectLoopRef.current);
        detectLoopRef.current = null;
      }
      
      // Let the canvas fade out via CSS transition instead of clearing immediately
      // if (canvasRef.current) {
      //   const ctx = canvasRef.current.getContext('2d');
      //   if (ctx) {
      //     ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      //   }
      // }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        setIsCameraActive(false);
      }
    }
  };

  const selectFaceForRegistration = (box: { x: number, y: number, w: number, h: number }) => {
    const video = videoRef.current;
    if (!video) return;

    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = 150;
    captureCanvas.height = 150;
    const cCtx = captureCanvas.getContext('2d');
    if (!cCtx) return;

    const sourceX = Math.max(0, box.x);
    const sourceY = Math.max(0, box.y);
    const sourceW = Math.min(video.videoWidth - sourceX, box.w);
    const sourceH = Math.min(video.videoHeight - sourceY, box.h);

    cCtx.drawImage(
      video,
      sourceX, sourceY, sourceW, sourceH,
      0, 0, 150, 150
    );

    const base64Image = captureCanvas.toDataURL('image/jpeg', 0.9);
    setSelectedFace({
      x: box.x,
      y: box.y,
      w: box.w,
      h: box.h,
      image: base64Image
    });
    setFirstName('');
    setLastName('');
    setDni('');
  };

  function captureCurrentFaceCrop(box: { x: number, y: number, w: number, h: number }): string | null {
    const video = videoRef.current;
    if (!video) return null;

    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = 150;
    captureCanvas.height = 150;
    const cCtx = captureCanvas.getContext('2d');
    if (!cCtx) return null;

    let targetBox = box;
    if (deepFaceBoxesRef.current && deepFaceBoxesRef.current.length > 0) {
      let closestBox = deepFaceBoxesRef.current[0];
      let minDist = Infinity;
      for (const b of deepFaceBoxesRef.current) {
        const dist = Math.hypot(b.x - box.x, b.y - box.y);
        if (dist < minDist) {
          minDist = dist;
          closestBox = b;
        }
      }
      if (minDist < 100) {
        targetBox = closestBox;
      }
    }

    const sourceX = Math.max(0, targetBox.x);
    const sourceY = Math.max(0, targetBox.y);
    const sourceW = Math.min(video.videoWidth - sourceX, targetBox.w);
    const sourceH = Math.min(video.videoHeight - sourceY, targetBox.h);

    cCtx.drawImage(
      video,
      sourceX, sourceY, sourceW, sourceH,
      0, 0, 150, 150
    );

    return captureCanvas.toDataURL('image/jpeg', 0.9);
  }

  function startInteractiveEnrollment() {
    if (!selectedFace) return;
    enrollmentPhotosRef.current = [];
    enrollmentStepRef.current = 'liveness';
    setEnrollmentStep('liveness');
    setRegistrationProgress(0);
    setRegistrationStatusText("PRUEBA DE VIDA: PARPADEA UNA VEZ PARA INICIAR...");
  }

  async function submitEnrollment(fName: string, lName: string, dniVal: string) {
    try {
      const res = await fetch('http://localhost:5000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          first_name: fName,
          last_name: lName,
          dni: dniVal,
          image: enrollmentPhotosRef.current
        })
      });
      const data = await res.json();
      if (data.success) {
        setRegistrationProgress(100);
        setRegistrationStatusText("¡REGISTRO COMPLETO! PERFIL PROCESADO CON ÉXITO");
        
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        setInfoMsg(`¡Rostro registrado con éxito como ${fName} ${lName}!`);
        setSelectedFace(null);
        setFirstName('');
        setLastName('');
        setDni('');
        fetchLogs();
      } else {
        setErrorMsg(data.error || 'No se pudo registrar el rostro.');
      }
    } catch (err) {
      console.error("Register face error:", err);
      setErrorMsg("No se pudo conectar al servidor para registrar el rostro.");
    } finally {
      setRegistrationProgress(null);
      setRegistrationStatusText('');
      setEnrollmentStep('idle');
      enrollmentStepRef.current = 'idle';
      setStatus('Conectado y Reproduciendo');
    }
  }

  async function cropFaceAndRegister(fName: string, lName: string, dniVal: string, face: { x: number, y: number, w: number, h: number, image: string }) {
    startInteractiveEnrollment();
  }

  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (analysisEngineRef.current !== 'deepface' || !isCameraActive) return;
    
    const container = e.currentTarget;
    const rect = container.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const containerWidth = rect.width;
    const containerHeight = rect.height;

    const video = videoRef.current;
    if (!video) return;
    const videoWidth = video.videoWidth || 640;
    const videoHeight = video.videoHeight || 480;

    const containerRatio = containerWidth / containerHeight;
    const videoRatio = videoWidth / videoHeight;

    let scale = 1;
    let offsetX = 0;
    let offsetY = 0;

    if (containerRatio > videoRatio) {
      scale = containerWidth / videoWidth;
      offsetY = (containerHeight - videoHeight * scale) / 2;
    } else {
      scale = containerHeight / videoHeight;
      offsetX = (containerWidth - videoWidth * scale) / 2;
    }

    const canvasX = (clickX - offsetX) / scale;
    const canvasY = (clickY - offsetY) / scale;

    const clickedBox = deepFaceBoxesRef.current.find(box => {
      return canvasX >= box.x && canvasX <= box.x + box.w &&
             canvasY >= box.y && canvasY <= box.y + box.h;
    });

    if (clickedBox && clickedBox.identity === 'Unknown') {
      selectFaceForRegistration(clickedBox);
    }
  };

  const registerFace = async (fName: string, lName: string, dniVal: string) => {
    const video = videoRef.current;
    if (!video) return;

    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = video.videoWidth || 640;
    captureCanvas.height = video.videoHeight || 480;
    const cCtx = captureCanvas.getContext('2d');
    if (!cCtx) return;

    cCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const base64Image = captureCanvas.toDataURL('image/jpeg', 0.85);

    setStatus('Registrando Rostro...');
    try {
      const res = await fetch('http://localhost:5000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          first_name: fName,
          last_name: lName,
          dni: dniVal,
          image: base64Image
        })
      });
      const data = await res.json();
      if (data.success) {
        setInfoMsg(`¡Rostro registrado con éxito como ${fName} ${lName}!`);
        setLoggedInUser(`${fName} ${lName}`);
        setLoggedInDni(dniVal);
        setIdentity(`${fName} ${lName}`);
        setIsRegistering(false);
        setFirstName('');
        setLastName('');
        setDni('');
        fetchLogs();
      } else {
        setErrorMsg(data.error || 'No se pudo registrar el rostro.');
      }
    } catch (err) {
      console.error("Register face error:", err);
      setErrorMsg("No se pudo conectar al servidor para registrar el rostro.");
    } finally {
      setStatus('Conectado y Reproduciendo');
    }
  };

  return (
    <div className="h-[100dvh] w-full bg-black text-white flex overflow-hidden font-mono relative">
      {/* Background Camera Feed */}
      <div className="absolute inset-0 z-0">
        {!isCameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white/50 z-10 font-mono text-sm">
            <Camera className="w-8 h-8 mb-4 opacity-50" />
            <p>SYSTEM.CAMERA_OFFLINE</p>
          </div>
        )}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`absolute inset-0 w-full h-full object-cover grayscale contrast-125 opacity-60 transition-opacity duration-500 ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
        />
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 w-full h-full object-cover pointer-events-none transition-opacity duration-500 z-[15] ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
        />
        {/* Vignette & Scanlines */}
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.8)_100%)] z-10" />
        <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(transparent_50%,rgba(0,0,0,0.25)_50%)] bg-[length:100%_4px] z-10" />
        
        {/* Decorative HUD Elements */}
        <div className="absolute inset-0 pointer-events-none z-10 flex items-center justify-center overflow-hidden">
          <div className="w-[150vw] h-[150vw] sm:w-[600px] sm:h-[600px] border border-white/10 rounded-full border-dashed animate-[spin_60s_linear_infinite] shrink-0" />
          <div className="absolute w-[100vw] h-[100vw] sm:w-[400px] sm:h-[400px] border border-white/5 rounded-full animate-[spin_40s_linear_infinite_reverse] shrink-0" />
          <div className="absolute w-px h-full bg-white/5" />
          <div className="absolute h-px w-full bg-white/5" />
        </div>
      </div>

      {/* Overlays */}
      <div 
        onClick={(e) => {
          const target = e.target as HTMLElement;
          if (target.closest('.hud-card') || target.closest('button') || target.closest('select') || target.closest('input')) {
            return;
          }
          handleCanvasClick(e);
        }}
        className="relative z-20 w-full h-full pointer-events-auto p-4 sm:p-6 overflow-y-auto overflow-x-hidden pb-32 sm:pb-6"
      >
        <div className="flex flex-col lg:flex-row justify-between gap-4 min-h-full">
          
          {/* Left Column */}
          <div className="contents lg:flex lg:flex-col lg:justify-between w-full lg:w-80 pointer-events-none shrink-0">
            
            {/* Top Left: System Status & Mobile Controls */}
            <div className="flex flex-col gap-4 shrink-0 order-1 lg:order-none pointer-events-auto">
              
              {/* Status */}
              <div className="hud-card flex flex-col items-start gap-4 shrink-0">
                <div className="flex items-start justify-between w-full">
                  <div>
                    <h1 className="text-2xl font-bold tracking-tighter text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.8)]">VISION_SYNC</h1>
                    <p className="text-[10px] text-white/70 font-mono uppercase tracking-widest mb-2">Lyria RealTime Engine v2.4</p>
                    
                    <div className="flex flex-col items-start gap-1 pointer-events-auto mt-2">
                      <span className="text-[8px] text-white/40 uppercase tracking-widest font-mono">Analysis Engine</span>
                      <div className="flex items-center gap-2">
                        <select
                          value={analysisEngine}
                          onChange={(e) => {
                            const val = e.target.value as 'mediapipe' | 'deepface';
                            setAnalysisEngine(val);
                            deepFaceBoxesRef.current = [];
                            setDemographics(null);
                          }}
                          className="bg-zinc-900/80 text-white/90 border border-white/20 px-2 py-1 text-[9px] font-mono rounded-none uppercase cursor-pointer hover:border-white/50 transition-colors focus:outline-none backdrop-blur-sm"
                        >
                          <option value="mediapipe">Local (MediaPipe)</option>
                          <option value="deepface">Server (DeepFace)</option>
                        </select>
                        {analysisEngine === 'deepface' && (
                          <span className={`text-[8px] uppercase tracking-wider font-mono ${deepFaceStatus === 'connected' ? 'text-green-400' : deepFaceStatus === 'connecting' ? 'text-yellow-400' : 'text-red-400'}`}>
                            [{deepFaceStatus}]
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {/* Small Start/Stop Button (Mobile Landscape Only) */}
                    <button
                      onClick={() => { playHoverSound(); isCameraActive ? stopSession(true) : startSession(); }}
                      onMouseEnter={playHoverSound}
                      disabled={!isModelLoaded}
                      className={`hidden landscape:flex lg:landscape:hidden justify-center items-center gap-2 px-3 py-2 font-mono text-[10px] font-bold uppercase tracking-widest transition-all duration-300 border backdrop-blur-md rounded-none ${
                        isCameraActive 
                          ? 'bg-red-500/20 text-red-400 border-red-500 hover:bg-red-500/30 shadow-[0_0_10px_rgba(239,68,68,0.4)]' 
                          : 'bg-white/10 text-white border-white hover:bg-white/20 shadow-[0_0_10px_rgba(255,255,255,0.3)]'
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      {!isModelLoaded ? (
                        <><Loader2 className="w-3 h-3 animate-spin" /> INIT</>
                      ) : isCameraActive ? (
                        <><Square className="w-3 h-3 fill-current" /> STOP</>
                      ) : (
                        <><Play className="w-3 h-3 fill-current" /> START</>
                      )}
                    </button>
                    <button 
                      onClick={() => { playHoverSound(); setIsInfoOpen(true); }}
                      onMouseEnter={playHoverSound}
                      className="p-2 bg-white/10 hover:bg-white/20 rounded-full transition-colors backdrop-blur-md border border-white/20 shrink-0"
                      title="Información de la Aplicación"
                    >
                      <Info className="w-5 h-5 text-white" />
                    </button>
                  </div>
                </div>
                <div className="text-xs font-mono text-white/80 flex items-center gap-2 bg-black/40 backdrop-blur px-3 py-1.5 border border-white/20">
                  <div className={`w-2 h-2 rounded-none ${status === 'Conectado y Reproduciendo' ? 'bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)]' : status.includes('Conectando') || status.includes('Iniciando') ? 'bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.8)]' : status === 'Cargando modelo de detección de objetos...' ? 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.8)]' : status.includes('Error') || status.includes('denegado') ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]' : 'bg-zinc-600'}`} />
                  {status}
                </div>
              </div>

              {/* Mobile Controls (Hidden on Desktop & Landscape) */}
              <div className="flex lg:hidden landscape:hidden flex-col items-stretch gap-4 shrink-0">
                <button
                  onClick={() => { playHoverSound(); isCameraActive ? stopSession(true) : startSession(); }}
                  onMouseEnter={playHoverSound}
                  disabled={!isModelLoaded}
                  className={`flex justify-center items-center gap-3 px-10 py-4 font-mono text-sm font-bold uppercase tracking-widest transition-all duration-300 border-2 backdrop-blur-md ${
                    isCameraActive 
                      ? 'bg-red-500/20 text-red-400 border-red-500 hover:bg-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.4)]' 
                      : 'bg-white/10 text-white border-white hover:bg-white/20 shadow-[0_0_20px_rgba(255,255,255,0.3)]'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  {!isModelLoaded ? (
                    <><Loader2 className="w-5 h-5 animate-spin" /> INITIALIZING...</>
                  ) : isCameraActive ? (
                    <><Square className="w-5 h-5 fill-current" /> STOP SYSTEM</>
                  ) : (
                    <><Play className="w-5 h-5 fill-current" /> START SYSTEM</>
                  )}
                </button>
              </div>

              {/* Mobile Camera Viewport Spacer */}
              <div className="h-[45vh] landscape:h-[100vh] lg:hidden pointer-events-none shrink-0" />
            </div>

            {/* Bottom Left: Scan & Affective */}
            <div className="flex flex-col landscape:flex-row lg:landscape:flex-col gap-4 shrink-0 lg:mt-auto order-4 lg:order-none pointer-events-auto">
              
              {/* Biometric Terminal HUD Card */}
              {analysisEngine === 'deepface' && (
                <div className="hud-card bg-black/40 backdrop-blur-md border border-white/20 p-4 w-full shadow-[0_0_30px_rgba(0,0,0,0.8)] relative flex flex-col shrink-0">
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-3 flex items-center gap-2">
                    <Key className="w-3 h-3 text-white/70" />
                    Terminal Biométrica
                  </h3>
                  
                  {/* Real-time Status Display */}
                  <div className="border border-white/10 bg-white/5 p-3 flex flex-col gap-2 relative overflow-hidden mb-3">
                    <div className="flex justify-between items-center">
                      <span className="text-[8px] text-white/40 uppercase tracking-widest font-mono">Estado del Sistema</span>
                      <span className="text-[9px] font-bold tracking-widest animate-pulse font-mono uppercase">
                        {!isCameraActive ? (
                          'CAMARA_APAGADA'
                        ) : !isFaceDetected ? (
                          <span className="text-blue-400">ESCANEANDO_ROSTRO...</span>
                        ) : loggedInUser ? (
                          <span className="text-green-400 drop-shadow-[0_0_5px_rgba(74,222,128,0.5)]">ACCESO_PERMITIDO</span>
                        ) : (
                          <span className="text-yellow-400">USUARIO_NO_REGISTRADO</span>
                        )}
                      </span>
                    </div>
                    
                    {/* Active Profile */}
                    <div className="flex flex-col mt-1">
                      <span className="text-[8px] text-white/30 uppercase tracking-widest">Perfil Activo</span>
                      <span className="text-xs font-bold text-white uppercase tracking-wider mt-0.5 truncate font-sans">
                        {loggedInUser ? loggedInUser : 'Ninguno'}
                      </span>
                      {loggedInUser && loggedInDni && (
                        <span className="text-[9px] text-white/50 uppercase tracking-wider font-mono mt-0.5">
                          DNI: {loggedInDni}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Rostros en Cámara */}
                  {isCameraActive && (
                    <div className="border border-white/10 bg-white/5 p-3 flex flex-col gap-2 mb-3">
                      <span className="text-[8px] text-white/40 uppercase tracking-widest font-mono">Rostros en Cámara</span>
                      {detectedFaces.length === 0 ? (
                        <p className="text-[10px] text-white/40 italic font-mono">No se detectan rostros.</p>
                      ) : (
                        <div className="flex flex-col gap-1.5 max-h-48 overflow-y-auto pr-1">
                          {detectedFaces.map((face, idx) => {
                            const isUnknown = face.identity === 'Unknown';
                            const isSelected = selectedFace && 
                              Math.abs(selectedFace.x - face.box.x) < 20 && 
                              Math.abs(selectedFace.y - face.box.y) < 20;
                              
                            return (
                              <div 
                                key={idx} 
                                className={`flex items-center justify-between p-1.5 border text-[10px] ${
                                  isSelected 
                                    ? 'border-blue-500 bg-blue-500/10' 
                                    : isUnknown 
                                      ? 'border-yellow-500/20 bg-yellow-500/5' 
                                      : 'border-green-500/20 bg-green-500/5'
                                }`}
                              >
                                <div className="flex flex-col min-w-0">
                                  <span className={`font-bold ${isUnknown ? 'text-yellow-400' : 'text-green-400'} truncate uppercase`}>
                                    {isUnknown ? 'Sujeto No Registrado' : face.identity}
                                  </span>
                                  <span className="text-[8px] text-white/60 font-mono">
                                    {isUnknown ? 'No Autorizado' : `DNI: ${face.dni || 'S/D'}`} | Edad: {face.age} | G: {face.gender.substring(0, 1)}
                                  </span>
                                </div>
                                {isUnknown && (
                                  <button
                                    onClick={() => selectFaceForRegistration(face.box)}
                                    className="px-2 py-0.5 border border-yellow-500 text-yellow-500 bg-transparent hover:bg-yellow-500 hover:text-black transition-all text-[8px] font-bold uppercase tracking-wider font-mono cursor-pointer"
                                  >
                                    {isSelected ? 'Reg...' : 'Reg'}
                                  </button>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Instructive Placeholder Helper */}
                  {!selectedFace && isCameraActive && detectedFaces.some(f => f.identity === 'Unknown') && (
                    <p className="text-[8px] text-yellow-500/80 font-mono text-center uppercase tracking-wide mb-2 animate-pulse">
                      [Haz clic en 'REG' o sobre un recuadro amarillo para asociar datos]
                    </p>
                  )}
                  
                  {/* Registration form if a face is selected */}
                  {selectedFace && (
                    <div className="border border-blue-500/30 bg-blue-500/5 p-3 flex flex-col gap-2 mb-3 relative">
                      <div className="flex justify-between items-center">
                        <span className="text-[9px] font-bold text-blue-400 uppercase tracking-wider font-mono">Registrar Rostro</span>
                        <button 
                          onClick={() => setSelectedFace(null)}
                          className="text-white/40 hover:text-white text-[9px] uppercase font-mono cursor-pointer"
                        >
                          Cancelar
                        </button>
                      </div>
                      
                      {/* Cropped Image Preview */}
                      <div className="flex gap-3 items-center my-1">
                        <div className="w-12 h-12 border border-white/20 bg-black overflow-hidden shrink-0 flex items-center justify-center">
                          <img src={selectedFace.image} alt="Crop" className="w-full h-full object-cover" />
                        </div>
                        <div className="text-[8px] text-white/50 font-mono">
                          Rostro capturado. Ingresa los datos para registrarlo.
                        </div>
                      </div>
                      
                      <div className="flex flex-col gap-2 mt-1">
                        <input
                          type="text"
                          placeholder="NOMBRE"
                          value={firstName}
                          onChange={(e) => setFirstName(e.target.value)}
                          className="bg-black/60 text-white border border-white/20 px-2 py-1 text-[10px] font-mono focus:outline-none focus:border-white/50 uppercase rounded-none"
                        />
                        <input
                          type="text"
                          placeholder="APELLIDO"
                          value={lastName}
                          onChange={(e) => setLastName(e.target.value)}
                          className="bg-black/60 text-white border border-white/20 px-2 py-1 text-[10px] font-mono focus:outline-none focus:border-white/50 uppercase rounded-none"
                        />
                        <input
                          type="text"
                          placeholder="DNI"
                          value={dni}
                          onChange={(e) => setDni(e.target.value)}
                          className="bg-black/60 text-white border border-white/20 px-2 py-1 text-[10px] font-mono focus:outline-none focus:border-white/50 uppercase rounded-none"
                        />
                        <button
                          onClick={() => {
                            if (firstName.trim() && lastName.trim() && dni.trim()) {
                              cropFaceAndRegister(firstName.trim(), lastName.trim(), dni.trim(), selectedFace);
                            } else {
                              setErrorMsg("Nombre, Apellido y DNI son obligatorios para el registro biométrico.");
                            }
                          }}
                          className="w-full mt-1 bg-blue-600 text-white text-[9px] font-bold uppercase tracking-widest py-1.5 hover:bg-blue-500 transition-colors rounded-none font-mono cursor-pointer"
                        >
                          Guardar Registro
                        </button>
                      </div>
                    </div>
                  )}
                  
                  {/* Logout option if logged in */}
                  {loggedInUser && (
                    <button
                      onClick={() => {
                        setLoggedInUser(null);
                        setLoggedInDni(null);
                        setIdentity('Unknown');
                        identityRef.current = 'Unknown';
                        setFirstName('');
                        setLastName('');
                        setDni('');
                      }}
                      className="w-full border border-red-500/30 text-red-400 bg-red-500/10 hover:bg-red-500/20 text-[9px] font-bold uppercase tracking-widest py-1.5 transition-colors rounded-none font-mono cursor-pointer"
                    >
                      Cerrar Sesión / Cambiar de Usuario
                    </button>
                  )}
                </div>
              )}
              
              {/* Biometric Access Logs Panel */}
              {analysisEngine === 'deepface' && (
                <div className="hud-card bg-black/40 backdrop-blur-md border border-white/20 p-4 w-full shadow-[0_0_30px_rgba(0,0,0,0.8)] relative flex flex-col h-48 shrink-0">
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2 flex items-center gap-2">
                    <Activity className="w-3.5 h-3.5 text-white/70" />
                    Registro de Accesos Biométricos
                  </h3>
                  <div className="flex-1 overflow-y-auto border border-white/10 bg-black/60 p-2 font-mono text-[9px] leading-tight space-y-1.5 scrollbar-thin scrollbar-thumb-white/20">
                    {accessLogs.length === 0 ? (
                      <div className="text-white/30 text-center py-4">SIN_REGISTROS</div>
                    ) : (
                      accessLogs.map((log) => {
                        const time = log.timestamp ? log.timestamp.split(' ')[1] || log.timestamp : '00:00:00';
                        return (
                          <div key={log.id} className="border-b border-white/5 pb-1 flex flex-col">
                            <div className="flex justify-between items-center">
                              <span className="text-white/40">[{time}]</span>
                              <span className={log.success ? "text-green-400 font-bold" : "text-yellow-500 font-bold"}>
                                {log.success ? "ACCESO_PERMITIDO" : "NO_AUTORIZADO"}
                              </span>
                            </div>
                            <div className="text-white/80 font-sans font-bold text-[10px] mt-0.5">
                              {log.name !== 'Unknown' ? log.name.toUpperCase() : 'SUJETO DESCONOCIDO'}
                            </div>
                            <div className="text-white/40 text-[8px] uppercase tracking-wider flex justify-between mt-0.5">
                              <span>Ánimo: {log.emotion}</span>
                              <span>Edad: ~{log.age}</span>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              )}
              
              {/* Middle Left: Face Scanner */}
              <div className="flex flex-col justify-center shrink-0 landscape:flex-1 lg:landscape:flex-none">
                <div className="bg-black/40 backdrop-blur-md border border-white/20 p-4 w-full shadow-[0_0_30px_rgba(0,0,0,0.8)] relative overflow-hidden flex flex-col h-64 landscape:h-full lg:landscape:h-64 shrink-0" title="Real-time facial landmark tracking">
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2 shrink-0 flex items-center gap-2">
                    <ScanFace className="w-3 h-3" />
                    Biometric Scan
                  </h3>
                  <div className="relative w-full flex-1 border border-white/10 flex items-center justify-center bg-white/5 min-h-0">
                    <canvas
                      ref={faceCanvasRef}
                      width={300}
                      height={300}
                      className={`w-full h-full object-contain transition-opacity duration-500 ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
                    />
                  </div>
                  {demographics && (
                    <div className="mt-2 text-[10px] text-white/70 uppercase tracking-wider flex justify-between bg-white/5 px-2 py-1 border border-white/10 shrink-0 font-mono">
                      <span>Age: ~{demographics.age}</span>
                      <span>Gender: {demographics.gender}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Bottom Left: Affective State */}
              <div className="flex flex-col justify-end shrink-0 landscape:flex-1 lg:landscape:flex-none">
                <div className="bg-black/40 backdrop-blur-md border border-white/20 p-5 w-full h-full shadow-[0_0_30px_rgba(0,0,0,0.8)]" title="Detected emotional state based on facial expressions">
                  <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-3 flex items-center gap-2">
                    <Activity className="w-3 h-3" />
                    Estado Afectivo
                  </h3>
                  <div className="text-3xl font-light tracking-tighter mb-4 capitalize text-white drop-shadow-[0_0_8px_rgba(255,255,255,0.5)]">
                    {translateEmotion(consoleState.emotion)}
                  </div>
                  
                  <div className="space-y-2">
                    {[
                      { label: 'Sonrisa', value: consoleState.blendshapes.smile },
                      { label: 'Ceño fruncido', value: consoleState.blendshapes.frown },
                      { label: 'Boca abierta', value: consoleState.blendshapes.mouthOpen },
                      { label: 'Cejas levantadas', value: consoleState.blendshapes.browRaise },
                      { label: 'Parpadeo', value: consoleState.blendshapes.eyeBlink },
                    ].map((item) => (
                      <div key={item.label}>
                        <div className="flex justify-between text-[10px] mb-1">
                          <span className="text-white/60 uppercase tracking-wider">{item.label}</span>
                          <span className="font-bold text-white/90">{isNaN(item.value) ? 0 : (item.value * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-[2px] bg-white/10 overflow-hidden">
                          <motion.div 
                            className="h-full bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)]"
                            initial={{ width: 0 }}
                            animate={{ width: `${isNaN(item.value) ? 0 : item.value * 100}%` }}
                            transition={{ type: 'spring', bounce: 0, duration: 0.5 }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

            </div>
          </div>

          {/* Right Column */}
          <div className="contents lg:flex lg:flex-col lg:justify-between lg:items-end w-full lg:w-80 pointer-events-none shrink-0 mt-0">
            
            {/* Top Right: Controls */}
            <div className="hidden lg:flex flex-col items-end gap-4 shrink-0 order-none pointer-events-auto">
              <button
                onClick={() => { playHoverSound(); isCameraActive ? stopSession(true) : startSession(); }}
                onMouseEnter={playHoverSound}
                disabled={!isModelLoaded}
                className={`flex justify-center items-center gap-3 px-10 py-4 font-mono text-sm font-bold uppercase tracking-widest transition-all duration-300 border-2 backdrop-blur-md ${
                  isCameraActive 
                    ? 'bg-red-500/20 text-red-400 border-red-500 hover:bg-red-500/30 shadow-[0_0_20px_rgba(239,68,68,0.4)]' 
                    : 'bg-white/10 text-white border-white hover:bg-white/20 shadow-[0_0_20px_rgba(255,255,255,0.3)]'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {!isModelLoaded ? (
                  <><Loader2 className="w-5 h-5 animate-spin" /> INITIALIZING...</>
                ) : isCameraActive ? (
                  <><Square className="w-5 h-5 fill-current" /> STOP SYSTEM</>
                ) : (
                  <><Play className="w-5 h-5 fill-current" /> START SYSTEM</>
                )}
              </button>
            </div>

            {/* Bottom Right: Entities & Audio */}
            <div className="flex flex-col gap-4 shrink-0 w-full order-2 lg:order-none pointer-events-auto">
              
              {/* Detected Entities */}
              <div className="hud-card order-1 lg:order-2 w-full bg-black/40 backdrop-blur-md border border-white/20 p-5 shadow-[0_0_30px_rgba(0,0,0,0.8)]" title="Objetos detectados en la vista de la cámara">
                <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-3 flex items-center gap-2">
                  <Cpu className="w-3 h-3" />
                  Entidades
                </h3>
                {consoleState.objects.length === 0 ? (
                  <p className="text-[10px] text-white/40 italic">No se detectaron entidades.</p>
                ) : (
                  <ul className="space-y-1.5">
                    <AnimatePresence>
                      {consoleState.objects.map((obj) => {
                        const translatedObj = obj === 'person' ? 'persona' :
                                              obj === 'cell phone' ? 'teléfono celular' :
                                              obj === 'laptop' ? 'computadora portátil' :
                                              obj === 'tv' ? 'televisor' :
                                              obj === 'cup' ? 'taza' :
                                              obj === 'bottle' ? 'botella' :
                                              obj === 'bowl' ? 'tazón' :
                                              obj === 'cat' ? 'gato' :
                                              obj === 'dog' ? 'perro' :
                                              obj === 'bird' ? 'pájaro' :
                                              obj === 'car' ? 'automóvil' :
                                              obj === 'bus' ? 'autobús' :
                                              obj === 'truck' ? 'camión' :
                                              obj === 'chair' ? 'silla' :
                                              obj === 'couch' ? 'sofá' :
                                              obj === 'bed' ? 'cama' :
                                              obj === 'potted plant' ? 'planta en maceta' :
                                              obj === 'book' ? 'libro' : obj;
                        return (
                          <motion.li 
                            key={obj}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 10 }}
                            className="text-[10px] flex items-center gap-2 text-white/90 uppercase tracking-wider"
                          >
                            <span className="w-1 h-1 bg-white shadow-[0_0_5px_rgba(255,255,255,0.8)]" />
                            {translatedObj}
                          </motion.li>
                        );
                      })}
                    </AnimatePresence>
                  </ul>
                )}
              </div>

              {/* Audio Profile */}
              <div className="hud-card order-2 lg:order-1 w-full bg-black/40 backdrop-blur-md border border-white/20 p-5 shadow-[0_0_30px_rgba(0,0,0,0.8)]" title="Prompt musical generado por IA basado en tu entorno y estado de ánimo">
                <h3 className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-3">Perfil de Audio</h3>
                <div className="relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-0.5 h-full bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)]" />
                  <p className="text-xs leading-relaxed text-white/90 pl-3">
                    {currentPrompt}
                  </p>
                </div>
              </div>

            </div>
          </div>

        </div>
      </div>
      {/* Error Modal */}
      <AnimatePresence>
        {errorMsg && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto"
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-zinc-900 border border-red-500/50 p-6 max-w-md w-full shadow-[0_0_40px_rgba(239,68,68,0.2)] relative"
            >
              <div className="flex items-start gap-4 mb-6">
                <div className="p-3 bg-red-500/10 border border-red-500/30 shrink-0">
                  <AlertCircle className="w-6 h-6 text-red-500" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-red-500 uppercase tracking-widest">{status}</h3>
                  <p className="text-sm mt-2 text-red-400/80 leading-relaxed">{errorMsg}</p>
                </div>
              </div>
              
              <button 
                onClick={() => setErrorMsg(null)}
                className={`w-full py-3 text-xs font-mono font-bold uppercase tracking-widest transition-colors ${
                  status === 'Error de Cámara' 
                    ? 'bg-red-500/20 hover:bg-red-500/30 border border-red-500/50 text-red-400' 
                    : 'bg-white/5 hover:bg-white/10 border border-white/20 text-white/70'
                }`}
              >
                Cerrar
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info Toast (Center Bottom) */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col justify-end items-center pointer-events-none z-30 w-[calc(100%-2rem)] sm:w-full max-w-md">
        <AnimatePresence>
          {infoMsg && !errorMsg && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="bg-black/80 backdrop-blur-md border border-white/30 p-4 flex items-start gap-3 text-white shadow-[0_0_20px_rgba(255,255,255,0.1)] mb-4 w-full"
            >
              <Music className="w-5 h-5 shrink-0 mt-0.5" />
              <div>
                <h3 className="font-bold text-sm">{status}</h3>
                <p className="text-xs mt-1 text-white/80">{infoMsg}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Info Modal */}
      <AnimatePresence>
        {isInfoOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto"
            onClick={() => setIsInfoOpen(false)}
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-zinc-900 border border-white/20 p-6 max-w-lg w-full shadow-[0_0_40px_rgba(0,0,0,0.8)] relative max-h-[90vh] overflow-y-auto"
            >
              <div className="flex flex-col-reverse sm:flex-row sm:items-start justify-between gap-4 mb-4">
                <h2 className="text-xl font-bold text-white flex items-center gap-2 self-start">
                  <Info className="w-5 h-5 shrink-0" />
                  Acerca de Vision to Music
                </h2>
                <button 
                  onClick={() => setIsInfoOpen(false)}
                  className="p-2 shrink-0 border border-white/20 bg-black/50 hover:bg-white/10 text-white/50 hover:text-white transition-colors self-end sm:self-auto"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              <div className="space-y-4 text-sm text-white/80 leading-relaxed">
                <p>
                  <strong>Vision to Music</strong> utiliza la cámara de tu dispositivo para analizar tus expresiones faciales y los objetos a tu alrededor en tiempo real.
                </p>
                <p>
                  Con base en estos datos visuales, genera un paisaje sonoro ambiental procedural y continuo que se adapta a tu estado de ánimo y entorno.
                </p>
                <ul className="list-disc pl-5 space-y-2 text-white/70">
                  <li><strong>Escaneo Biométrico:</strong> Realiza un seguimiento de los puntos de tu rostro para determinar tu emoción actual (feliz, triste, sorprendido, enojado, miedo, disgustado).</li>
                  <li><strong>Entidades:</strong> Detecta objetos en tu entorno (como computadoras portátiles, tazas, plantas) para influir en la vibración musical.</li>
                  <li><strong>Perfil de Audio:</strong> La IA genera un prompt descriptivo basado en la escena, el cual impulsa el motor de música procedural.</li>
                </ul>
                <p className="text-xs text-white/50 mt-4 pt-4 border-t border-white/10">
                  Nota: Todo el procesamiento ocurre de forma local en tu navegador o mediante llamadas seguras a la API. No se guarda ni se transmite ningún dato de video.
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Registration Progress HUD Overlay */}
      {registrationProgress !== null && (
        <div className="fixed top-4 right-4 sm:top-6 sm:right-6 lg:top-28 z-[100] font-mono pointer-events-auto">
          <div className="w-80 bg-zinc-950/90 border border-blue-500/50 p-5 shadow-[0_0_35px_rgba(59,130,246,0.25)] backdrop-blur-md flex flex-col items-center text-center">
            <Loader2 className="w-10 h-10 text-blue-500 animate-spin mb-4" />
            <span className="text-xs uppercase tracking-widest text-blue-400 font-bold mb-2">Procesando Registro</span>
            
            {/* Percentage and bar */}
            <span className="text-3xl font-light text-white mb-4 drop-shadow-[0_0_10px_rgba(255,255,255,0.4)]">
              {registrationProgress}%
            </span>
            
            <div className="w-full h-[3px] bg-white/10 mb-4 overflow-hidden relative">
              <div 
                className="h-full bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.8)] transition-all duration-300"
                style={{ width: `${registrationProgress}%` }}
              />
            </div>
            
            <span className="text-[10px] text-white/70 uppercase tracking-wider h-12 text-center px-2 flex items-center justify-center mb-4">
              {registrationStatusText}
            </span>

            {/* Quality Gauges for Interactive Steps */}
            {enrollmentStep !== 'idle' && enrollmentStep !== 'liveness' && enrollmentStep !== 'submitting' && (
              <div className="w-full border-t border-white/10 pt-4 flex flex-col gap-2 text-[8px] text-white/60 text-left font-mono">
                <div className="flex justify-between">
                  <span>BRILLO (60 - 210):</span>
                  <span className={hudQuality.brightness < 60 || hudQuality.brightness > 215 ? "text-yellow-500 font-bold animate-pulse" : "text-green-400 font-bold"}>
                    {hudQuality.brightness.toFixed(0)} ({hudQuality.brightness < 60 ? 'BAJO' : hudQuality.brightness > 215 ? 'ALTO' : 'OK'})
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>EXPRESIÓN NEUTRAL:</span>
                  <span className={hudQuality.isNeutral ? "text-green-400 font-bold" : "text-yellow-500 font-bold animate-pulse"}>
                    {hudQuality.isNeutral ? 'SÍ' : 'NO NEUTRAL'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>ÁNGULO DE GIRO (YAW):</span>
                  <span className="text-white font-bold">
                    {hudPose.yaw > 8 ? 'DERECHA' : hudPose.yaw < -8 ? 'IZQUIERDA' : 'CENTRO'} ({hudPose.yaw.toFixed(0)}°)
                  </span>
                </div>
                
                {hudQuality.warning && (
                  <div className="mt-2 p-1.5 bg-yellow-500/10 border border-yellow-500/30 text-yellow-500 font-bold uppercase tracking-wider text-center text-[7px] animate-pulse">
                    ⚠️ {hudQuality.warning}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
