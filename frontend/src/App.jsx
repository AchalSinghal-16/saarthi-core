import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';
import {
  Activity,
  AlertTriangle,
  Camera,
  CheckCircle2,
  Cpu,
  MapPin,
  Mic,
  MicOff,
  Navigation,
  Phone,
  Radio,
  RefreshCw,
  RotateCw,
  Send,
  Settings,
  ShieldAlert,
  Sparkles,
  StopCircle,
  Thermometer,
  UploadCloud,
  UserPlus,
  Users,
  Volume2,
  VolumeX,
  Wifi,
  X,
  BookOpen,
  Banknote,
  Eye,
  Sliders,
  ChevronRight,
} from 'lucide-react';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

function ChangeView({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center && center[0] && center[1]) {
      map.setView(center, map.getZoom());
    }
  }, [center, map]);
  return null;
}

const API_BASE = 'http://localhost:8000';

export default function App() {
  // Navigation & Active Panel State
  const [activeTab, setActiveTab] = useState('faces'); // 'faces' | 'emergency' | 'camera'

  // Face Enrollment State
  const [faceName, setFaceName] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [enrollLoading, setEnrollLoading] = useState(false);
  const [enrollStatus, setEnrollStatus] = useState(null);

  // Enrolled Faces State
  const [enrolledFaces, setEnrolledFaces] = useState([]);
  const [facesLoading, setFacesLoading] = useState(false);

  // Emergency Contact State
  const [emergencyPhone, setEmergencyPhone] = useState('');
  const [emergencyStatus, setEmergencyStatus] = useState(null);
  const [emergencyLoading, setEmergencyLoading] = useState(false);

  // Camera Source State (Phone IP stream or webcam)
  const [cameraSource, setCameraSource] = useState('0');
  const [cameraStatus, setCameraStatus] = useState(null);
  const [cameraLoading, setCameraLoading] = useState(false);

  // Live Location State (Ola Maps & Browser GPS)
  const [locationData, setLocationData] = useState(null);
  const [locationLoading, setLocationLoading] = useState(false);

  // System Vitals State (CPU & Temperature)
  const [vitals, setVitals] = useState({ cpu_percent: 0, cpu_temp: 45.0 });

  // Caregiver Intercom State (Push-to-Talk)
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [intercomStatus, setIntercomStatus] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordTimerRef = useRef(null);
  const startTimeRef = useRef(0);

  // Active Voice Mode (Ask VLM) State
  const [isVoiceListening, setIsVoiceListening] = useState(false);
  const [voiceSeconds, setVoiceSeconds] = useState(0);
  const voiceRecorderRef = useRef(null);
  const voiceChunksRef = useRef([]);
  const voiceTimerRef = useRef(null);

  // Assistive Action Trigger State
  const [actionStatus, setActionStatus] = useState(null);

  // Trigger Assistive Feature via REST API
  const triggerAction = async (actionType, loadingMsg) => {
    setActionStatus(loadingMsg);
    try {
      const res = await fetch(`${API_BASE}/api/trigger/${actionType}`, { method: 'POST' });
      const data = await res.json();
      if (data.success && data.result) {
        if (actionType === 'currency') {
          setActionStatus(`💵 ${data.result}`);
        } else if (actionType === 'read') {
          setActionStatus(`📖 ${data.result}`);
        } else {
          setActionStatus(`${data.result}`);
        }
      } else {
        setActionStatus(`Triggered ${actionType}.`);
      }
    } catch (err) {
      console.error(`Failed to trigger ${actionType}:`, err);
      setActionStatus(`Trigger error: ${err.message}`);
    }
  };

  // Remote Audio Kill-Switch: Halt all audio on wearable
  const stopAudio = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/audio/stop`, { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        setActionStatus('🔇 Audio stopped immediately.');
      }
    } catch (err) {
      console.error('Failed to stop audio:', err);
      setActionStatus(`Audio stop error: ${err.message}`);
    }
  };

  // Sync real-time browser/device GPS to backend
  const syncBrowserLocation = (onSuccess = null) => {
    if (!navigator.geolocation) {
      console.warn('Geolocation is not supported by this browser.');
      return;
    }
    setActionStatus('📍 Acquiring device GPS coordinates...');
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude } = pos.coords;
        console.log(`[Browser GPS] Acquired exact coordinates: ${latitude}, ${longitude}`);
        try {
          const res = await fetch(`${API_BASE}/api/location/coordinates`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat: latitude, lng: longitude, source: 'Browser GPS' }),
          });
          const data = await res.json();
          if (data.success) {
            setLocationData({
              success: true,
              lat: latitude,
              lng: longitude,
              coordinates: { lat: latitude, lng: longitude },
              address: data.address,
              source: 'Browser GPS',
            });
            setActionStatus(`📍 Location synced: ${data.address || `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`}`);
            if (onSuccess) onSuccess({ latitude, longitude, address: data.address });
          }
        } catch (err) {
          console.error('Failed to sync coordinates to backend:', err);
          setActionStatus(`Location sync error: ${err.message}`);
        }
      },
      (err) => {
        console.warn('[Browser GPS] Error:', err.message);
        setActionStatus(`GPS permission: ${err.message}. Please allow location access in your browser.`);
      },
      { enableHighAccuracy: true, timeout: 8000, maximumAge: 30000 }
    );
  };

  // Trigger Location with browser GPS coordinates passed directly
  const triggerLocationAction = async () => {
    setActionStatus('📍 Pinpointing exact location via GPS & Ola Maps...');
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        async (pos) => {
          const { latitude, longitude } = pos.coords;
          try {
            const res = await fetch(`${API_BASE}/api/trigger/location`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ lat: latitude, lng: longitude }),
            });
            const data = await res.json();
            if (data.success && data.result) {
              setActionStatus(data.result);
              fetchLocation();
            }
          } catch (err) {
            triggerAction('location', '📍 Announcing Ola Maps street address...');
          }
        },
        () => {
          triggerAction('location', '📍 Announcing Ola Maps street address...');
        },
        { enableHighAccuracy: true, timeout: 5000 }
      );
    } else {
      triggerAction('location', '📍 Announcing Ola Maps street address...');
    }
  };

  // Active Voice Mode: Record voice directly from browser or phone
  const toggleVoiceQuery = async () => {
    if (isVoiceListening) {
      if (voiceRecorderRef.current && voiceRecorderRef.current.state !== 'inactive') {
        voiceRecorderRef.current.stop();
      }
      return;
    }

    try {
      setActionStatus('🎙️ Accessing microphone for voice query...');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      voiceChunksRef.current = [];

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : '';
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          voiceChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        if (voiceTimerRef.current) clearInterval(voiceTimerRef.current);
        setIsVoiceListening(false);
        setVoiceSeconds(0);

        const audioBlob = new Blob(voiceChunksRef.current, { type: 'audio/webm' });
        if (audioBlob.size < 200) {
          setActionStatus('Voice query too brief. Please speak clearly into the microphone.');
          return;
        }

        setActionStatus('🤔 Asking Gemini VLM... Please wait.');
        const formData = new FormData();
        formData.append('file', audioBlob, 'query.webm');

        try {
          const res = await fetch(`${API_BASE}/api/trigger/voice`, {
            method: 'POST',
            body: formData,
          });
          const data = await res.json();
          if (data.success && data.result) {
            setActionStatus(`${data.result}`);
          } else {
            setActionStatus('Could not complete voice query.');
          }
        } catch (err) {
          setActionStatus(`Voice query error: ${err.message}`);
        }
      };

      recorder.start(100);
      voiceRecorderRef.current = recorder;
      setIsVoiceListening(true);
      setVoiceSeconds(0);
      setActionStatus('🎙️ Listening to you! Speak your question now (Click Ask VLM again to submit)...');

      let sec = 0;
      voiceTimerRef.current = setInterval(() => {
        sec += 1;
        setVoiceSeconds(sec);
        if (sec >= 7) {
          if (voiceRecorderRef.current && voiceRecorderRef.current.state !== 'inactive') {
            voiceRecorderRef.current.stop();
          }
        }
      }, 1000);
    } catch (err) {
      console.error('Microphone access denied:', err);
      setActionStatus(`Microphone error: ${err.message}. Triggering wearable mic...`);
      try {
        const res = await fetch(`${API_BASE}/api/trigger/voice`, { method: 'POST' });
        const data = await res.json();
        if (data.success && data.result) setActionStatus(`${data.result}`);
      } catch (fallbackErr) {
        setActionStatus(`Trigger error: ${fallbackErr.message}`);
      }
    }
  };

  // Intercom: Start recording caregiver microphone
  const startIntercom = async () => {
    if (isRecording) return;
    try {
      setIntercomStatus({ type: 'info', message: 'Requesting microphone access...' });
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : '';
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        if (recordTimerRef.current) clearInterval(recordTimerRef.current);

        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        if (audioBlob.size < 200) {
          setIntercomStatus({
            type: 'error',
            message: 'Audio recording too brief. Please speak for at least 1-2 seconds.',
          });
          setIsRecording(false);
          setRecordingSeconds(0);
          return;
        }

        const sizeKb = (audioBlob.size / 1024).toFixed(1);
        setIntercomStatus({ type: 'info', message: `Transmitting audio (${sizeKb} KB) to wearable...` });

        const formData = new FormData();
        formData.append('file', audioBlob, 'temp_intercom.webm');

        try {
          const res = await fetch(`${API_BASE}/api/intercom`, {
            method: 'POST',
            body: formData,
          });
          const data = await res.json();
          if (data.success) {
            setIntercomStatus({ type: 'success', message: 'Transmission played in user earbuds!' });
          } else {
            setIntercomStatus({ type: 'error', message: data.message || 'Intercom transmission failed.' });
          }
        } catch (err) {
          setIntercomStatus({ type: 'error', message: `Transmission error: ${err.message}` });
        } finally {
          setIsRecording(false);
          setRecordingSeconds(0);
        }
      };

      recorder.start(100);
      mediaRecorderRef.current = recorder;
      startTimeRef.current = Date.now();
      setIsRecording(true);
      setRecordingSeconds(0);
      setIntercomStatus(null);

      recordTimerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Microphone access denied:', err);
      setIntercomStatus({
        type: 'error',
        message: 'Microphone permission blocked. Please allow mic access in your browser address bar.',
      });
      setIsRecording(false);
    }
  };

  const stopIntercom = () => {
    if (recordTimerRef.current) clearInterval(recordTimerRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop();
      } catch (err) {
        console.error('Error stopping recorder:', err);
      }
    }
  };

  const cancelIntercom = () => {
    if (recordTimerRef.current) clearInterval(recordTimerRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.onstop = null;
        mediaRecorderRef.current.stop();
      } catch (err) {
        console.error('Error cancelling recorder:', err);
      }
    }
    audioChunksRef.current = [];
    setIsRecording(false);
    setRecordingSeconds(0);
    setIntercomStatus({ type: 'info', message: 'Recording cancelled.' });
  };

  const toggleIntercom = () => {
    if (isRecording) {
      stopIntercom();
    } else {
      startIntercom();
    }
  };

  const fetchVitals = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/vitals`);
      const data = await res.json();
      if (data.success) {
        setVitals({
          cpu_percent: typeof data.cpu_percent === 'number' ? data.cpu_percent : 0,
          cpu_temp: typeof data.cpu_temp === 'number' ? data.cpu_temp : 45.0,
        });
      }
    } catch (err) {
      console.error('Failed to fetch vitals:', err);
    }
  };

  const fetchLocation = async () => {
    setLocationLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/location`);
      const data = await res.json();
      if (data.success) {
        setLocationData(data);
      }
    } catch (err) {
      console.error('Failed to fetch location:', err);
    } finally {
      setLocationLoading(false);
    }
  };

  const fetchCameraSource = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/settings/source`);
      const data = await res.json();
      if (data.success && data.source) {
        setCameraSource(data.source);
      }
    } catch (err) {
      console.error('Failed to fetch camera source:', err);
    }
  };

  const fetchFaces = async () => {
    setFacesLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/faces`);
      const data = await res.json();
      if (data.success && data.faces) {
        setEnrolledFaces(data.faces);
      }
    } catch (err) {
      console.error('Failed to fetch faces:', err);
    } finally {
      setFacesLoading(false);
    }
  };

  const fetchEmergencySettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/settings/emergency`);
      const data = await res.json();
      if (data.success && data.settings?.phone_number) {
        setEmergencyPhone(data.settings.phone_number);
      }
    } catch (err) {
      console.error('Failed to fetch emergency settings:', err);
    }
  };

  useEffect(() => {
    fetchFaces();
    fetchEmergencySettings();
    fetchCameraSource();
    fetchLocation();
    syncBrowserLocation();
    fetchVitals();

    const vitalsInterval = setInterval(fetchVitals, 5000);
    const locationInterval = setInterval(fetchLocation, 10000);

    const handleKeyDown = (e) => {
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
      const key = e.key.toLowerCase();
      if (key === 'r') {
        e.preventDefault();
        triggerAction('read', '📖 Reading document text...');
      } else if (key === 'c') {
        e.preventDefault();
        triggerAction('currency', '💵 Scanning Indian Rupee banknote...');
      } else if (key === 'v') {
        e.preventDefault();
        toggleVoiceQuery();
      } else if (key === 'l') {
        e.preventDefault();
        triggerLocationAction();
      } else if (key === 'x') {
        e.preventDefault();
        stopAudio();
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      clearInterval(vitalsInterval);
      clearInterval(locationInterval);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setFilePreview(URL.createObjectURL(file));
      setEnrollStatus(null);
    }
  };

  const handleEnrollSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile || !faceName.trim()) {
      setEnrollStatus({ type: 'error', message: 'Please provide both a portrait photo and full name.' });
      return;
    }

    setEnrollLoading(true);
    setEnrollStatus(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('name', faceName.trim());

    try {
      const res = await fetch(`${API_BASE}/api/faces/enroll`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (res.ok && data.success) {
        setEnrollStatus({ type: 'success', message: data.message || `Enrolled ${faceName} into ChromaDB successfully!` });
        setFaceName('');
        setSelectedFile(null);
        setFilePreview(null);
        fetchFaces();
      } else {
        setEnrollStatus({ type: 'error', message: data.message || 'Face enrollment failed. Please try a clearer portrait.' });
      }
    } catch (err) {
      setEnrollStatus({ type: 'error', message: `Server error: ${err.message}` });
    } finally {
      setEnrollLoading(false);
    }
  };

  const handleEmergencySubmit = async (e) => {
    e.preventDefault();
    if (!emergencyPhone.trim()) return;

    setEmergencyLoading(true);
    setEmergencyStatus(null);

    try {
      const res = await fetch(`${API_BASE}/api/settings/emergency`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone_number: emergencyPhone.trim() }),
      });
      const data = await res.json();

      if (res.ok && data.success) {
        setEmergencyStatus({ type: 'success', message: 'Emergency contact saved to permanent storage!' });
      } else {
        setEmergencyStatus({ type: 'error', message: data.detail || 'Failed to update emergency contact.' });
      }
    } catch (err) {
      setEmergencyStatus({ type: 'error', message: `Connection error: ${err.message}` });
    } finally {
      setEmergencyLoading(false);
    }
  };

  const handleCameraSourceSubmit = async (e) => {
    e.preventDefault();
    if (!cameraSource.trim()) return;

    setCameraLoading(true);
    setCameraStatus(null);

    try {
      const res = await fetch(`${API_BASE}/api/settings/source`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: cameraSource.trim() }),
      });
      const data = await res.json();

      if (res.ok && data.success) {
        setCameraStatus({ type: 'success', message: data.message });
      } else {
        setCameraStatus({ type: 'error', message: data.message || 'Could not connect to camera source.' });
      }
    } catch (err) {
      setCameraStatus({ type: 'error', message: `Connection error: ${err.message}` });
    } finally {
      setCameraLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      
      {/* ── Enterprise Master Navbar ── */}
      <header className="sticky top-0 z-50 border-b border-slate-800/80 bg-[#090e1a]/90 backdrop-blur-xl px-5 lg:px-8 py-3.5 shadow-2xl">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
          
          {/* Logo & Platform Info */}
          <div className="flex items-center gap-3.5">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-500 to-amber-400 p-[1.5px] shadow-lg shadow-cyan-500/20">
                <div className="w-full h-full bg-[#090e1a] rounded-[10px] flex items-center justify-center font-extrabold text-white text-lg tracking-wider">
                  S
                </div>
              </div>
              <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-500 border-2 border-[#090e1a] rounded-full"></span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                  Saarthi
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-cyan-950/80 text-cyan-400 border border-cyan-800/80 font-bold tracking-wider">
                    OS 2.0 • Caregiver Portal
                  </span>
                </h1>
              </div>
              <p className="text-[11px] text-slate-400 font-medium">Edge AI Assistive Wearable Telemetry & Biometrics</p>
            </div>
          </div>

          {/* Real-Time Hardware Telemetry Bar */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* CPU Badge */}
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900/80 border border-slate-800 text-xs font-mono shadow-inner">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-slate-400">CPU:</span>
              <span className="text-cyan-300 font-bold">{vitals.cpu_percent.toFixed(0)}%</span>
            </div>

            {/* Core Temp Badge */}
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900/80 border border-slate-800 text-xs font-mono shadow-inner">
              <Thermometer className={`w-3.5 h-3.5 ${vitals.cpu_temp > 65 ? 'text-rose-400' : 'text-emerald-400'}`} />
              <span className="text-slate-400">Core:</span>
              <span className={`font-bold ${vitals.cpu_temp > 65 ? 'text-rose-400' : 'text-emerald-300'}`}>
                {vitals.cpu_temp.toFixed(0)}°C
              </span>
            </div>

            {/* Live Telemetry Status */}
            <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-emerald-950/60 border border-emerald-800/80 text-emerald-300 text-xs font-semibold shadow-sm">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="hidden md:inline">Edge Stream Active (15 FPS)</span>
              <span className="md:hidden">Live</span>
            </div>
          </div>
        </div>
      </header>

      {/* ── Main Dashboard Workspace ── */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 lg:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* ════════ LEFT COLUMN: Live Perception & Geospatial Radar (7 Cols) ════════ */}
        <div className="lg:col-span-7 flex flex-col gap-5">
          
          {/* Card: Live Edge Camera Telemetry */}
          <div className="bg-[#0b1222] border border-slate-800/90 rounded-2xl overflow-hidden shadow-2xl flex flex-col ring-1 ring-white/5">
            
            {/* Video Header */}
            <div className="px-5 py-3 border-b border-slate-800/80 flex items-center justify-between bg-slate-900/50">
              <div className="flex items-center gap-2.5">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-rose-500"></span>
                </span>
                <h2 className="font-semibold text-xs uppercase tracking-wider text-slate-200 flex items-center gap-2">
                  <Camera className="w-4 h-4 text-cyan-400" />
                  Live Wearable Perception Feed
                </h2>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-mono text-slate-400 px-2.5 py-0.5 rounded-md bg-slate-950 border border-slate-800">
                  YOLOv8 + FaceNet • 15 FPS
                </span>
              </div>
            </div>

            {/* Video Viewer */}
            <div className="relative aspect-video w-full bg-black flex items-center justify-center overflow-hidden group">
              <img
                src={`${API_BASE}/api/video_feed`}
                alt="Saarthi Live Telemetry Stream"
                className="w-full h-full object-contain"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = 'https://placehold.co/640x480/090e1a/94a3b8?text=Connecting+to+Wearable+Stream...';
                }}
              />

              {/* Real-time HUD Status Watermark */}
              <div className="absolute top-3 left-3 bg-slate-950/80 backdrop-blur-md border border-slate-800/90 px-3 py-1.5 rounded-lg text-xs font-mono text-slate-300 pointer-events-none flex items-center gap-2 shadow-lg">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                <span>Spatial HUD Active</span>
              </div>

              {/* Source Watermark */}
              <div className="absolute top-3 right-3 bg-slate-950/80 backdrop-blur-md border border-slate-800/90 px-2.5 py-1 rounded-md text-[10px] font-mono text-slate-400 pointer-events-none">
                SRC: {cameraSource}
              </div>
            </div>

            {/* Spatial Hazard Detection Legend */}
            <div className="p-3.5 bg-slate-950/70 border-t border-slate-800/80 grid grid-cols-3 gap-2.5 text-center text-xs">
              <div className="bg-slate-900/60 p-2 rounded-xl border border-emerald-900/40 flex flex-col items-center">
                <span className="text-[10px] uppercase font-bold text-slate-400">Safe Sector</span>
                <span className="font-semibold text-emerald-400 text-xs flex items-center gap-1 mt-0.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-400"></span> &gt; 2.5m Obstacle
                </span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded-xl border border-rose-900/40 flex flex-col items-center">
                <span className="text-[10px] uppercase font-bold text-slate-400">Hazard Zone</span>
                <span className="font-semibold text-rose-400 text-xs flex items-center gap-1 mt-0.5">
                  <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span> &lt; 2.5m Proximity
                </span>
              </div>
              <div className="bg-slate-900/60 p-2 rounded-xl border border-amber-900/40 flex flex-col items-center">
                <span className="text-[10px] uppercase font-bold text-slate-400">Biometrics</span>
                <span className="font-semibold text-amber-400 text-xs flex items-center gap-1 mt-0.5">
                  <span className="w-2 h-2 rounded-full bg-amber-400"></span> Face ID Speech Alert
                </span>
              </div>
            </div>
          </div>

          {/* Card: Real-Time Geospatial Map & Ola Maps Address */}
          <div className="bg-[#0b1222] border border-slate-800/90 rounded-2xl overflow-hidden shadow-2xl flex flex-col ring-1 ring-white/5">
            
            {/* Map Header with Real-Time GPS Sync */}
            <div className="px-5 py-3 border-b border-slate-800/80 flex items-center justify-between bg-slate-900/50 flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <Navigation className="w-4 h-4 text-emerald-400" />
                <h3 className="font-semibold text-xs uppercase tracking-wider text-slate-200">
                  Wearable Geospatial Radar
                </h3>
              </div>
              
              <div className="flex items-center gap-2">
                {locationData?.source && (
                  <span className={`text-[10px] px-2.5 py-0.5 rounded-full font-semibold border ${
                    locationData.source.includes('GPS')
                      ? 'bg-emerald-950/80 text-emerald-300 border-emerald-700/80'
                      : 'bg-amber-950/80 text-amber-300 border-amber-700/80'
                  }`}>
                    {locationData.source}
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => syncBrowserLocation()}
                  className="text-xs px-3 py-1 rounded-lg bg-emerald-950 hover:bg-emerald-900 border border-emerald-700/80 text-emerald-200 font-semibold transition flex items-center gap-1.5 shadow-sm active:scale-95"
                  title="Acquire high-precision browser GPS coordinates"
                >
                  <MapPin className="w-3.5 h-3.5" /> Sync Device GPS
                </button>
              </div>
            </div>

            {/* Interactive Leaflet Radar */}
            <div className="p-3 bg-slate-950 flex flex-col gap-2.5">
              <div className="h-64 w-full rounded-xl overflow-hidden border border-slate-800 relative z-0 shadow-inner">
                <MapContainer
                  center={[
                    locationData?.coordinates?.lat || locationData?.lat || 28.6139,
                    locationData?.coordinates?.lng || locationData?.lng || 77.2090,
                  ]}
                  zoom={15}
                  scrollWheelZoom={false}
                  style={{ height: '100%', width: '100%' }}
                >
                  <ChangeView
                    center={[
                      locationData?.coordinates?.lat || locationData?.lat || 28.6139,
                      locationData?.coordinates?.lng || locationData?.lng || 77.2090,
                    ]}
                  />
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  <Marker
                    position={[
                      locationData?.coordinates?.lat || locationData?.lat || 28.6139,
                      locationData?.coordinates?.lng || locationData?.lng || 77.2090,
                    ]}
                  >
                    <Popup>
                      <div className="text-xs font-sans text-slate-900 font-semibold p-1">
                        <strong className="text-emerald-700">Wearable Position</strong><br />
                        {locationData?.address || 'Live GPS Tracked'}
                      </div>
                    </Popup>
                  </Marker>
                </MapContainer>
              </div>

              {/* Ola Maps Geocoded Street Banner */}
              <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-800 flex items-center justify-between gap-3 text-xs">
                <div className="flex items-center gap-2 truncate max-w-[75%]">
                  <span className="p-1.5 rounded-lg bg-emerald-950/80 border border-emerald-800 text-emerald-400 shrink-0">
                    <MapPin className="w-3.5 h-3.5" />
                  </span>
                  <div className="truncate">
                    <span className="text-[10px] text-slate-400 block font-mono">Resolved Street Address (Ola Maps):</span>
                    <span className="font-medium text-slate-200 truncate block">
                      {locationData?.address || 'Live GPS Position (Press [L] to announce street address)'}
                    </span>
                  </div>
                </div>

                <div className="text-right shrink-0">
                  <span className="text-[10px] text-slate-400 block font-mono">GPS Coordinates:</span>
                  <span className="font-mono text-xs text-amber-400 font-bold">
                    {(locationData?.coordinates?.lat || locationData?.lat || 28.6139)?.toFixed(4)}, {(locationData?.coordinates?.lng || locationData?.lng || 77.2090)?.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ════════ RIGHT COLUMN: Control Hub & Caregiver Operations (5 Cols) ════════ */}
        <div className="lg:col-span-5 flex flex-col gap-5">

          {/* ── CARD 1: Instant Wearable Assistive Commands & Kill-Switch ── */}
          <div className="bg-[#0b1222] border border-slate-800/90 rounded-2xl p-4 shadow-xl flex flex-col gap-3.5 ring-1 ring-white/5">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-amber-400" />
                <h3 className="font-bold text-xs uppercase tracking-wider text-slate-200">
                  Wearable Assistive Controls
                </h3>
              </div>
              <span className="text-[10px] text-cyan-400 font-mono">Hotkeys: [R, C, V, L, X]</span>
            </div>

            {/* Action Grid (4 Actions) */}
            <div className="grid grid-cols-2 gap-2.5">
              
              {/* Read Document Text */}
              <button
                type="button"
                onClick={() => triggerAction('read', '📖 Reading document text with Gemini VLM...')}
                className="py-2.5 px-3 rounded-xl bg-slate-900/90 hover:bg-slate-800/90 border border-cyan-800/60 hover:border-cyan-500 text-cyan-300 text-xs font-semibold flex items-center justify-between transition-all group shadow-sm active:scale-[0.98]"
              >
                <div className="flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-cyan-400 group-hover:scale-110 transition-transform" />
                  <span>Read Text</span>
                </div>
                <kbd className="text-[10px] bg-slate-950 px-1.5 py-0.5 rounded text-slate-400 font-mono border border-slate-800">R</kbd>
              </button>

              {/* Currency Recognition */}
              <button
                type="button"
                onClick={() => triggerAction('currency', '💵 Scanning Indian Rupee banknote...')}
                className="py-2.5 px-3 rounded-xl bg-slate-900/90 hover:bg-slate-800/90 border border-amber-800/60 hover:border-amber-500 text-amber-300 text-xs font-semibold flex items-center justify-between transition-all group shadow-sm active:scale-[0.98]"
              >
                <div className="flex items-center gap-2">
                  <Banknote className="w-4 h-4 text-amber-400 group-hover:scale-110 transition-transform" />
                  <span>Currency</span>
                </div>
                <kbd className="text-[10px] bg-slate-950 px-1.5 py-0.5 rounded text-slate-400 font-mono border border-slate-800">C</kbd>
              </button>

              {/* Active Voice Mode (Ask VLM) */}
              <button
                type="button"
                onClick={toggleVoiceQuery}
                className={`py-2.5 px-3 rounded-xl border text-xs font-semibold flex items-center justify-between transition-all group shadow-sm active:scale-[0.98] ${
                  isVoiceListening
                    ? 'bg-rose-950 border-rose-500 text-rose-200 animate-pulse'
                    : 'bg-slate-900/90 hover:bg-slate-800/90 border-violet-800/60 hover:border-violet-500 text-violet-300'
                }`}
              >
                <div className="flex items-center gap-2">
                  {isVoiceListening ? (
                    <Mic className="w-4 h-4 text-rose-400 animate-bounce" />
                  ) : (
                    <Sparkles className="w-4 h-4 text-violet-400 group-hover:scale-110 transition-transform" />
                  )}
                  <span>{isVoiceListening ? `Listening 0:0${voiceSeconds}` : 'Ask VLM'}</span>
                </div>
                <kbd className="text-[10px] bg-slate-950 px-1.5 py-0.5 rounded text-slate-400 font-mono border border-slate-800">V</kbd>
              </button>

              {/* Announce Street Location */}
              <button
                type="button"
                onClick={triggerLocationAction}
                className="py-2.5 px-3 rounded-xl bg-slate-900/90 hover:bg-slate-800/90 border border-emerald-800/60 hover:border-emerald-500 text-emerald-300 text-xs font-semibold flex items-center justify-between transition-all group shadow-sm active:scale-[0.98]"
              >
                <div className="flex items-center gap-2">
                  <Navigation className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition-transform" />
                  <span>Location</span>
                </div>
                <kbd className="text-[10px] bg-slate-950 px-1.5 py-0.5 rounded text-slate-400 font-mono border border-slate-800">L</kbd>
              </button>
            </div>

            {/* Remote Audio Kill-Switch Button */}
            <button
              type="button"
              onClick={stopAudio}
              className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-red-950/90 to-red-900/90 hover:from-red-900 hover:to-red-800 border border-red-700/80 text-red-200 text-xs font-bold flex items-center justify-between transition-all active:scale-[0.99] shadow-lg shadow-red-950/40"
            >
              <div className="flex items-center gap-2">
                <VolumeX className="w-4 h-4 text-red-300" />
                <span>Immediate Audio Kill-Switch (Halt Speech)</span>
              </div>
              <kbd className="text-[10px] bg-red-950 px-2 py-0.5 rounded font-mono border border-red-700 text-red-300">X</kbd>
            </button>

            {/* Live Result / Status Feedback Toast */}
            {actionStatus && (
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-200 flex items-center justify-between gap-2 shadow-inner">
                <span className="truncate max-w-[88%] font-medium">{actionStatus}</span>
                <button
                  type="button"
                  onClick={() => setActionStatus(null)}
                  className="text-slate-500 hover:text-slate-300 p-1"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>

          {/* ── CARD 2: Caregiver Studio Intercom (Two-Way Communication) ── */}
          <div className="bg-[#0b1222] border border-slate-800/90 rounded-2xl p-4 shadow-xl flex flex-col gap-3 ring-1 ring-white/5">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
              <div className="flex items-center gap-2">
                <Radio className="w-4 h-4 text-rose-400" />
                <h3 className="font-bold text-xs uppercase tracking-wider text-slate-200">
                  Caregiver Earbud Intercom
                </h3>
              </div>
              <span className={`text-[10px] px-2.5 py-0.5 rounded-full font-mono font-bold border ${
                isRecording
                  ? 'bg-rose-950 text-rose-300 border-rose-700 animate-pulse'
                  : 'bg-slate-950 text-slate-400 border-slate-800'
              }`}>
                {isRecording ? `TRANSMITTING 0:0${recordingSeconds}` : 'INTERCOM READY'}
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              Click once to record your message. Click again to transmit instantly to the user's wearable earbuds.
            </p>

            <div className="flex flex-col gap-2">
              <button
                type="button"
                onClick={toggleIntercom}
                className={`w-full py-3.5 px-4 rounded-xl font-bold text-xs select-none transition-all duration-200 flex items-center justify-center gap-2.5 shadow-lg active:scale-[0.98] ${
                  isRecording
                    ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-600/50 ring-4 ring-rose-500/30 animate-pulse'
                    : 'bg-gradient-to-r from-rose-950 via-slate-900 to-rose-950 hover:from-rose-900 hover:to-rose-900 text-rose-200 border border-rose-800/70 hover:border-rose-600'
                }`}
              >
                {isRecording ? (
                  <>
                    <StopCircle className="w-4 h-4 animate-bounce" />
                    <span>REC 0:0{recordingSeconds} — Click to Send Transmission</span>
                  </>
                ) : (
                  <>
                    <Mic className="w-4 h-4 text-rose-400" />
                    <span>Click to Record Voice Message</span>
                  </>
                )}
              </button>

              {isRecording && (
                <button
                  type="button"
                  onClick={cancelIntercom}
                  className="py-1 px-3 text-xs text-slate-400 hover:text-slate-200 self-center underline font-medium"
                >
                  Cancel transmission
                </button>
              )}
            </div>

            {intercomStatus && (
              <div className={`p-2.5 rounded-xl text-xs font-medium border text-center ${
                intercomStatus.type === 'success'
                  ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                  : intercomStatus.type === 'info'
                  ? 'bg-cyan-950/60 border-cyan-800 text-cyan-300 animate-pulse'
                  : 'bg-rose-950/60 border-rose-800 text-rose-300'
              }`}>
                {intercomStatus.message}
              </div>
            )}
          </div>

          {/* ── CARD 3: Administrative Hub (Segmented Clean Tabs) ── */}
          <div className="bg-[#0b1222] border border-slate-800/90 rounded-2xl p-4 shadow-xl flex flex-col gap-4 ring-1 ring-white/5">
            
            {/* Tab Bar Header */}
            <div className="flex items-center gap-1.5 p-1 bg-slate-950 rounded-xl border border-slate-800/80">
              <button
                type="button"
                onClick={() => setActiveTab('faces')}
                className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                  activeTab === 'faces'
                    ? 'bg-slate-800 text-amber-300 shadow-sm border border-slate-700'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Users className="w-3.5 h-3.5" />
                <span>Biometrics ({enrolledFaces.length})</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab('emergency')}
                className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                  activeTab === 'emergency'
                    ? 'bg-slate-800 text-rose-300 shadow-sm border border-slate-700'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <ShieldAlert className="w-3.5 h-3.5" />
                <span>Emergency SOS</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab('camera')}
                className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                  activeTab === 'camera'
                    ? 'bg-slate-800 text-cyan-300 shadow-sm border border-slate-700'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Sliders className="w-3.5 h-3.5" />
                <span>Camera Stream</span>
              </button>
            </div>

            {/* ── TAB PANEL 1: Biometric Face ID & Directory ── */}
            {activeTab === 'faces' && (
              <div className="flex flex-col gap-4">
                
                {/* Face Enrollment Form */}
                <form onSubmit={handleEnrollSubmit} className="flex flex-col gap-3">
                  <div>
                    <label className="block text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1">
                      Person's Full Name
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. Dr. Sharma, Mom, Rahul"
                      value={faceName}
                      onChange={(e) => setFaceName(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500/50"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1">
                      Portrait Photograph
                    </label>
                    <input
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      onChange={handleFileChange}
                      className="w-full text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-amber-400 hover:file:bg-slate-700 cursor-pointer"
                      required
                    />
                  </div>

                  {/* Photo Preview */}
                  {filePreview && (
                    <div className="flex items-center gap-3 p-2.5 bg-slate-950 rounded-xl border border-slate-800">
                      <img src={filePreview} alt="Preview" className="w-12 h-12 rounded-lg object-cover border border-amber-500/50 shadow" />
                      <div className="text-xs truncate">
                        <span className="font-semibold text-slate-200 block truncate">{selectedFile?.name}</span>
                        <span className="text-[11px] text-slate-500">{(selectedFile?.size / 1024).toFixed(1)} KB • Ready for FaceNet</span>
                      </div>
                    </div>
                  )}

                  {enrollStatus && (
                    <div className={`p-2.5 rounded-xl text-xs font-medium border ${
                      enrollStatus.type === 'success'
                        ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                        : 'bg-rose-950/60 border-rose-800 text-rose-300'
                    }`}>
                      {enrollStatus.message}
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={enrollLoading}
                    className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 text-slate-950 font-bold text-xs shadow-md shadow-amber-500/20 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                  >
                    {enrollLoading ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        Extracting FaceNet Vector...
                      </>
                    ) : (
                      <>
                        <UserPlus className="w-4 h-4" />
                        Enroll Face into ChromaDB
                      </>
                    )}
                  </button>
                </form>

                {/* Enrolled Contacts Directory */}
                <div className="border-t border-slate-800/80 pt-3 flex flex-col gap-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                      Enrolled Contacts ({enrolledFaces.length})
                    </span>
                    <button
                      type="button"
                      onClick={fetchFaces}
                      disabled={facesLoading}
                      className="text-[11px] text-amber-400 hover:text-amber-300 flex items-center gap-1 font-semibold"
                    >
                      <RefreshCw className={`w-3 h-3 ${facesLoading ? 'animate-spin' : ''}`} /> Refresh
                    </button>
                  </div>

                  <div className="max-h-36 overflow-y-auto pr-1 flex flex-col gap-1.5 custom-scrollbar">
                    {enrolledFaces.length === 0 ? (
                      <p className="text-xs text-slate-500 py-3 text-center italic">No faces enrolled in ChromaDB yet.</p>
                    ) : (
                      enrolledFaces.map((f, i) => (
                        <div key={i} className="flex items-center justify-between px-3 py-2 bg-slate-950/60 rounded-xl border border-slate-800/70 text-xs">
                          <div className="flex items-center gap-2">
                            <span className="w-5 h-5 rounded-full bg-amber-950 text-amber-300 border border-amber-800/80 flex items-center justify-center font-bold text-[10px]">
                              {f.name.charAt(0).toUpperCase()}
                            </span>
                            <span className="font-medium text-slate-200">{f.name}</span>
                          </div>
                          <span className="text-[10px] font-mono text-slate-500">{f.id}</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>

              </div>
            )}

            {/* ── TAB PANEL 2: Emergency SOS Configuration ── */}
            {activeTab === 'emergency' && (
              <form onSubmit={handleEmergencySubmit} className="flex flex-col gap-3.5">
                <div>
                  <label className="block text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1">
                    Primary Caregiver Alert Number
                  </label>
                  <div className="relative">
                    <Phone className="w-4 h-4 text-rose-400 absolute left-3 top-3" />
                    <input
                      type="tel"
                      placeholder="+91 98765 43210"
                      value={emergencyPhone}
                      onChange={(e) => setEmergencyPhone(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-3.5 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-rose-500/50"
                      required
                    />
                  </div>
                  <span className="text-[11px] text-slate-500 mt-1 block">
                    Saved permanently to disk in <code className="text-rose-400">backend/settings.json</code>.
                  </span>
                </div>

                {emergencyStatus && (
                  <div className={`p-2.5 rounded-xl text-xs font-medium border ${
                    emergencyStatus.type === 'success'
                      ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                      : 'bg-rose-950/60 border-rose-800 text-rose-300'
                  }`}>
                    {emergencyStatus.message}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={emergencyLoading}
                  className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-rose-600 to-rose-700 hover:from-rose-500 hover:to-rose-600 text-white font-bold text-xs shadow-md shadow-rose-600/20 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                >
                  {emergencyLoading ? 'Persisting to Storage...' : 'Save Emergency Contact Permanently'}
                </button>
              </form>
            )}

            {/* ── TAB PANEL 3: Camera Stream Link Switcher ── */}
            {activeTab === 'camera' && (
              <form onSubmit={handleCameraSourceSubmit} className="flex flex-col gap-3.5">
                <div>
                  <label className="block text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1">
                    Active Video Source URL or Device Index
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. http://192.168.1.15:8080/video or 0"
                    value={cameraSource}
                    onChange={(e) => setCameraSource(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2 text-xs font-mono text-cyan-300 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
                    required
                  />
                  <span className="text-[11px] text-slate-500 mt-1 block">
                    Use <code className="text-cyan-400">0</code> for laptop webcam, or phone IP Webcam stream URL.
                  </span>
                </div>

                {cameraStatus && (
                  <div className={`p-2.5 rounded-xl text-xs font-medium border ${
                    cameraStatus.type === 'success'
                      ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                      : 'bg-rose-950/60 border-rose-800 text-rose-300'
                  }`}>
                    {cameraStatus.message}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={cameraLoading}
                  className="w-full py-2.5 px-4 rounded-xl bg-cyan-950 hover:bg-cyan-900 text-cyan-300 border border-cyan-800 font-bold text-xs shadow-sm transition flex items-center justify-center gap-2"
                >
                  {cameraLoading ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Connecting Stream...
                    </>
                  ) : (
                    'Switch Video Source'
                  )}
                </button>
              </form>
            )}

          </div>

        </div>
      </main>
    </div>
  );
}
