import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

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

  // Live Location State (Ola Maps)
  const [locationData, setLocationData] = useState(null);
  const [locationLoading, setLocationLoading] = useState(false);

  // System Vitals State (CPU & Temperature)
  const [vitals, setVitals] = useState({ cpu_percent: 0, cpu_temp: 45.0 });

  // Caregiver Intercom State (Push-to-Talk & Click-to-Talk)
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
        setActionStatus(`${data.result}`);
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

      // Collect data slices every 100ms
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

  // Intercom: Stop recording and upload
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
        mediaRecorderRef.current.onstop = null; // Detach upload handler
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

  // Fetch vitals
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

  // Fetch location
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

  // Fetch camera source
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

  // Fetch enrolled faces
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

  // Fetch emergency settings
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
    syncBrowserLocation(); // Automatically acquire real-time browser/device GPS
    fetchVitals();

    const vitalsInterval = setInterval(fetchVitals, 5000);
    const locationInterval = setInterval(fetchLocation, 10000);

    // Browser-level keyboard shortcut triggers (works whenever dashboard is open)
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

  // Handle file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setFilePreview(URL.createObjectURL(file));
      setEnrollStatus(null);
    }
  };

  // Submit face enrollment
  const handleEnrollSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile || !faceName.trim()) {
      setEnrollStatus({ type: 'error', message: 'Please provide both a photo and a name.' });
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
        setEnrollStatus({ type: 'success', message: data.message || `Enrolled ${faceName} successfully!` });
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

  // Submit emergency contact update
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
        setEmergencyStatus({ type: 'success', message: 'Emergency contact updated successfully!' });
      } else {
        setEmergencyStatus({ type: 'error', message: data.detail || 'Failed to update emergency contact.' });
      }
    } catch (err) {
      setEmergencyStatus({ type: 'error', message: `Connection error: ${err.message}` });
    } finally {
      setEmergencyLoading(false);
    }
  };

  // Submit camera source update (e.g. phone IP webcam)
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
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur px-6 py-4 flex items-center justify-between shadow-lg">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-cyan-400 flex items-center justify-center font-bold text-slate-950 text-xl shadow-md">
            S
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              Saarthi <span className="text-xs uppercase px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800 font-semibold tracking-wider">Caregiver Hub</span>
            </h1>
            <p className="text-xs text-slate-400">Assistive Wearable Remote Telemetry & Biometrics</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Quick-glance Vitals Badge */}
          <div className="hidden sm:flex items-center gap-2.5 px-3 py-1.5 rounded-lg bg-slate-950/70 border border-slate-800 text-xs font-mono">
            <span className="text-slate-400">CPU:</span>
            <span className="text-cyan-400 font-semibold">{vitals.cpu_percent.toFixed(0)}%</span>
            <span className="text-slate-700">•</span>
            <span className="text-slate-400">Temp:</span>
            <span className={`font-semibold ${vitals.cpu_temp > 65 ? 'text-rose-400' : 'text-emerald-400'}`}>
              {vitals.cpu_temp.toFixed(0)}°C
            </span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-950/60 border border-emerald-800/80 text-emerald-400 text-xs font-medium">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            Telemetry Stream Live
          </div>
        </div>
      </header>

      {/* Main Two-Column Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Live Video Telemetry Feed (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl flex flex-col">
            <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-ping"></span>
                <h2 className="font-semibold text-sm text-slate-200">Wearable Edge Camera Telemetry</h2>
              </div>
              <span className="text-xs text-slate-400 font-mono">MJPEG • 30 FPS • YOLOv8 + FaceNet</span>
            </div>

            {/* Video Container */}
            <div className="relative aspect-video w-full bg-slate-950 flex items-center justify-center overflow-hidden group">
              <img
                src={`${API_BASE}/api/video_feed`}
                alt="Saarthi Live Telemetry Stream"
                className="w-full h-full object-contain"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = 'https://placehold.co/640x480/0f172a/94a3b8?text=Connecting+to+Saarthi+Stream...';
                }}
              />
              <div className="absolute bottom-3 left-3 bg-slate-950/80 backdrop-blur border border-slate-800 px-3 py-1.5 rounded-lg text-xs text-slate-300 pointer-events-none flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                Live Spatial Distance HUD Active
              </div>
            </div>

            {/* Telemetry Guide Info */}
            <div className="p-4 bg-slate-900/40 border-t border-slate-800/80 grid grid-cols-3 gap-3 text-center text-xs">
              <div className="bg-slate-950/50 p-2.5 rounded-xl border border-slate-800/50">
                <span className="block text-slate-400">Green Boxes</span>
                <span className="font-semibold text-emerald-400">Safe Objects (&gt;2.5m)</span>
              </div>
              <div className="bg-slate-950/50 p-2.5 rounded-xl border border-slate-800/50">
                <span className="block text-slate-400">Red Boxes</span>
                <span className="font-semibold text-rose-400">Hazard Zone (&lt;2.5m)</span>
              </div>
              <div className="bg-slate-950/50 p-2.5 rounded-xl border border-slate-800/50">
                <span className="block text-slate-400">Face Recognition</span>
                <span className="font-semibold text-amber-400">Auto Name Audio Alert</span>
              </div>
            </div>
          </div>

          {/* Live GPS Map Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl flex flex-col">
            <div className="px-5 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-900/60 flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                <h3 className="font-semibold text-sm text-slate-200 flex items-center gap-2">
                  <span>🗺️</span> Real-Time GPS Tracking Map
                </h3>
              </div>
              <div className="flex items-center gap-2">
                {locationData?.source && (
                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                    locationData.source.includes('GPS') ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : 'bg-amber-950 text-amber-300 border border-amber-800'
                  }`}>
                    {locationData.source}
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => syncBrowserLocation()}
                  className="text-[11px] px-2.5 py-1 rounded-lg bg-emerald-950 hover:bg-emerald-900 border border-emerald-700/80 text-emerald-200 font-semibold transition flex items-center gap-1 shadow-sm"
                  title="Acquire real device GPS coordinates from browser"
                >
                  <span>📍</span> Sync Device GPS
                </button>
              </div>
            </div>

            <div className="p-3 bg-slate-950 flex flex-col gap-2">
              <div className="h-64 w-full rounded-xl overflow-hidden border border-slate-800 relative z-0">
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
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  <Marker
                    position={[
                      locationData?.coordinates?.lat || locationData?.lat || 28.6139,
                      locationData?.coordinates?.lng || locationData?.lng || 77.2090,
                    ]}
                  >
                    <Popup>
                      <div className="text-xs font-sans text-slate-900 font-medium">
                        <strong>Wearable Position</strong><br />
                        {locationData?.address || 'Live GPS Tracked'}
                      </div>
                    </Popup>
                  </Marker>
                </MapContainer>
              </div>

              {locationData && (
                <div className="text-xs text-slate-300 px-2 py-1 flex items-center justify-between">
                  <span className="text-slate-400 truncate max-w-[70%]">
                    📍 {locationData?.address || 'Live GPS Position (Press [L] for Street Name)'}
                  </span>
                  <span className="font-mono text-[10px] text-amber-400">
                    {(locationData?.coordinates?.lat || locationData?.lat)?.toFixed(4)}, {(locationData?.coordinates?.lng || locationData?.lng)?.toFixed(4)}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Biometrics & Caregiver Controls (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6">

          {/* Card: Wearable System Vitals */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col gap-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
              <div className="flex items-center gap-2">
                <span className="text-violet-400 text-sm">⚡</span>
                <h3 className="font-bold text-xs uppercase tracking-wider text-slate-200">Wearable System Vitals</h3>
              </div>
              <span className="text-[10px] text-slate-500 font-mono">Auto-refreshes (5s)</span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              {/* CPU Load */}
              <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3 flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-slate-400 font-medium">CPU Usage</span>
                  <span className="text-xs font-mono font-bold text-cyan-400">
                    {vitals.cpu_percent.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-cyan-400 h-1.5 rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(100, Math.max(2, vitals.cpu_percent))}%` }}
                  ></div>
                </div>
              </div>

              {/* CPU Temperature */}
              <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3 flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-slate-400 font-medium">Core Temp</span>
                  <span className={`text-xs font-mono font-bold ${vitals.cpu_temp > 65 ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {vitals.cpu_temp.toFixed(1)}°C
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className={`h-1.5 rounded-full transition-all duration-500 ${vitals.cpu_temp > 65 ? 'bg-rose-400' : 'bg-emerald-400'}`}
                    style={{ width: `${Math.min(100, (vitals.cpu_temp / 85) * 100)}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          {/* Card: Instant Wearable Assistive Actions */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl flex flex-col gap-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
              <div className="flex items-center gap-2">
                <span className="text-amber-400">⚡</span>
                <h3 className="font-bold text-xs uppercase tracking-wider text-slate-200">Wearable Assistive Controls</h3>
              </div>
              <span className="text-[10px] text-cyan-400 font-mono">Click or press [R, C, V, L]</span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => triggerAction('read', '📖 Reading document text with Gemini VLM...')}
                className="py-2.5 px-3 rounded-xl bg-slate-950 hover:bg-slate-800 border border-cyan-800/60 text-cyan-300 text-xs font-semibold flex items-center justify-center gap-2 transition hover:scale-[1.02]"
              >
                <span>📖</span> Read Text <kbd className="text-[10px] bg-slate-800 px-1 rounded text-slate-400">R</kbd>
              </button>
              <button
                type="button"
                onClick={() => triggerAction('currency', '💵 Scanning Indian Rupee banknote...')}
                className="py-2.5 px-3 rounded-xl bg-slate-950 hover:bg-slate-800 border border-amber-800/60 text-amber-300 text-xs font-semibold flex items-center justify-center gap-2 transition hover:scale-[1.02]"
              >
                <span>💵</span> Currency <kbd className="text-[10px] bg-slate-800 px-1 rounded text-slate-400">C</kbd>
              </button>
              <button
                type="button"
                onClick={toggleVoiceQuery}
                className={`py-2.5 px-3 rounded-xl border text-xs font-semibold flex items-center justify-center gap-2 transition hover:scale-[1.02] ${
                  isVoiceListening
                    ? 'bg-rose-950 border-rose-500 text-rose-200 animate-pulse'
                    : 'bg-slate-950 hover:bg-slate-800 border-pink-800/60 text-pink-300'
                }`}
              >
                <span>{isVoiceListening ? '🛑' : '🎙️'}</span>
                {isVoiceListening ? `Listening 00:0${voiceSeconds} (Click to Submit)` : 'Ask VLM'}{' '}
                <kbd className="text-[10px] bg-slate-800 px-1 rounded text-slate-400">V</kbd>
              </button>
              <button
                type="button"
                onClick={triggerLocationAction}
                className="py-2.5 px-3 rounded-xl bg-slate-950 hover:bg-slate-800 border border-emerald-800/60 text-emerald-300 text-xs font-semibold flex items-center justify-center gap-2 transition hover:scale-[1.02]"
              >
                <span>📍</span> Location <kbd className="text-[10px] bg-slate-800 px-1 rounded text-slate-400">L</kbd>
              </button>
            </div>

            {/* Remote Audio Kill-Switch Button */}
            <button
              type="button"
              onClick={stopAudio}
              className="w-full py-2.5 px-3 rounded-xl bg-red-950/80 hover:bg-red-900 border border-red-700/80 text-red-200 text-xs font-bold flex items-center justify-center gap-2 transition hover:scale-[1.01] shadow-md shadow-red-950/50"
            >
              <span className="text-sm font-black">⏹️</span> Stop Audio (✕) <kbd className="text-[10px] bg-red-900/90 px-1.5 py-0.5 rounded text-red-200 font-mono">X</kbd>
            </button>

            {actionStatus && (
              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 flex items-center justify-between">
                <span className="truncate max-w-[85%]">{actionStatus}</span>
                <button type="button" onClick={() => setActionStatus(null)} className="text-slate-500 hover:text-slate-300 text-xs">✕</button>
              </div>
            )}
          </div>

          {/* Card: Caregiver Intercom (Push-to-Talk) */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
              <h3 className="font-bold text-xs uppercase tracking-wider text-slate-200 flex items-center gap-2">
                <span className="text-rose-400">🎙️</span> Caregiver Intercom (Push-to-Talk)
              </h3>
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold border ${
                isRecording
                  ? 'bg-rose-950 text-rose-300 border-rose-700 animate-pulse'
                  : 'bg-slate-950 text-slate-400 border-slate-800'
              }`}>
                {isRecording ? `REC 00:0${recordingSeconds}` : 'READY'}
              </span>
            </div>

            <p className="text-xs text-slate-400">
              Click the button once to start recording, speak your message, then click again to send to user earbuds.
            </p>

            <div className="flex flex-col gap-2">
              <button
                type="button"
                onClick={toggleIntercom}
                className={`w-full py-4 px-4 rounded-xl font-bold text-sm select-none transition-all duration-200 flex items-center justify-center gap-3 shadow-lg ${
                  isRecording
                    ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-600/50 scale-[0.98] ring-4 ring-rose-500/40 animate-pulse'
                    : 'bg-gradient-to-r from-rose-900/80 via-slate-800 to-rose-900/80 hover:from-rose-800 hover:to-rose-800 text-rose-200 border border-rose-800/80 hover:scale-[1.01]'
                }`}
              >
                <span className={`text-xl ${isRecording ? 'animate-bounce' : ''}`}>
                  {isRecording ? '🛑' : '🎙️'}
                </span>
                <span>
                  {isRecording
                    ? `REC 00:0${recordingSeconds} — Click to Send Transmission`
                    : 'Click to Record Voice Message'}
                </span>
              </button>

              {isRecording && (
                <button
                  type="button"
                  onClick={cancelIntercom}
                  className="py-1 px-3 text-xs text-slate-400 hover:text-slate-200 self-center underline"
                >
                  Cancel recording
                </button>
              )}
            </div>

            {intercomStatus && (
              <div className={`p-2.5 rounded-lg text-xs font-medium border text-center ${
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

          {/* Card 1: Enroll New Face */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="font-bold text-base text-slate-100 flex items-center gap-2">
                  <span className="text-amber-400">👤</span> Biometric Face Enrollment
                </h3>
                <p className="text-xs text-slate-400">Register familiar contacts into wearable Face Memory</p>
              </div>
            </div>

            <form onSubmit={handleEnrollSubmit} className="flex flex-col gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Person's Full Name
                </label>
                <input
                  type="text"
                  placeholder="e.g. John Doe, Mom, Dr. Sharma"
                  value={faceName}
                  onChange={(e) => setFaceName(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500"
                  required
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Face Portrait Photo
                </label>
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleFileChange}
                  className="w-full text-xs text-slate-400 file:mr-3 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-amber-400 hover:file:bg-slate-700 cursor-pointer"
                  required
                />
              </div>

              {/* Photo Preview */}
              {filePreview && (
                <div className="flex items-center gap-3 p-3 bg-slate-950/80 rounded-xl border border-slate-800">
                  <img src={filePreview} alt="Preview" className="w-14 h-14 rounded-lg object-cover border border-amber-500/50 shadow" />
                  <div className="text-xs">
                    <span className="font-semibold text-slate-200 block truncate max-w-[200px]">{selectedFile?.name}</span>
                    <span className="text-slate-500">{(selectedFile?.size / 1024).toFixed(1)} KB • Ready for FaceNet</span>
                  </div>
                </div>
              )}

              {/* Status Message */}
              {enrollStatus && (
                <div className={`p-3 rounded-xl text-xs font-medium border ${
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
                className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 text-slate-950 font-bold text-sm shadow-lg shadow-amber-500/20 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
              >
                {enrollLoading ? (
                  <>
                    <span className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin"></span>
                    Extracting Embeddings...
                  </>
                ) : (
                  'Enroll Face into ChromaDB'
                )}
              </button>
            </form>
          </div>

          {/* Card 2: Registered Faces List */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
              <h3 className="font-bold text-sm text-slate-200 flex items-center gap-2">
                <span>📋</span> Enrolled Contacts ({enrolledFaces.length})
              </h3>
              <button
                onClick={fetchFaces}
                disabled={facesLoading}
                className="text-xs text-amber-400 hover:text-amber-300 flex items-center gap-1 font-medium transition"
              >
                {facesLoading ? 'Refreshing...' : '↻ Refresh'}
              </button>
            </div>

            <div className="max-h-40 overflow-y-auto pr-1 flex flex-col gap-1.5">
              {enrolledFaces.length === 0 ? (
                <p className="text-xs text-slate-500 py-3 text-center italic">No faces enrolled yet. Use the form above.</p>
              ) : (
                enrolledFaces.map((f, i) => (
                  <div key={i} className="flex items-center justify-between px-3 py-2 bg-slate-950/50 rounded-xl border border-slate-800/60 text-xs">
                    <span className="font-medium text-slate-200">{f.name}</span>
                    <span className="text-[10px] font-mono text-slate-500">{f.id}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Card 3: Emergency SOS Configuration */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-3">
            <h3 className="font-bold text-sm text-slate-200 flex items-center gap-2 border-b border-slate-800 pb-2.5">
              <span className="text-rose-400">🚨</span> Caregiver Emergency SOS Settings
            </h3>
            
            <form onSubmit={handleEmergencySubmit} className="flex flex-col gap-3">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Primary Alert Phone Number
                </label>
                <input
                  type="tel"
                  placeholder="+91 98765 43210"
                  value={emergencyPhone}
                  onChange={(e) => setEmergencyPhone(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-rose-500/50"
                  required
                />
              </div>

              {emergencyStatus && (
                <div className={`p-2.5 rounded-lg text-xs font-medium border ${
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
                className="py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs border border-slate-700 transition"
              >
                {emergencyLoading ? 'Updating...' : 'Save Emergency Contact'}
              </button>
            </form>
          </div>

          {/* Card 4: Wearable Camera Source Link (Phone / Webcam) */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-3">
            <h3 className="font-bold text-sm text-slate-200 flex items-center gap-2 border-b border-slate-800 pb-2.5">
              <span className="text-cyan-400">📱</span> Wearable Camera Stream Link
            </h3>
            
            <form onSubmit={handleCameraSourceSubmit} className="flex flex-col gap-3">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Video Stream URL or Camera Index
                </label>
                <input
                  type="text"
                  placeholder="e.g. http://192.168.1.10:8080/video or 0"
                  value={cameraSource}
                  onChange={(e) => setCameraSource(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs font-mono text-cyan-300 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
                  required
                />
                <span className="text-[11px] text-slate-500 mt-1 block">
                  Tip: Use <code className="text-cyan-400">http://&lt;phone_ip&gt;:8080/video</code> for Android IP Webcam, or <code className="text-cyan-400">0</code> for webcam.
                </span>
              </div>

              {cameraStatus && (
                <div className={`p-2.5 rounded-lg text-xs font-medium border ${
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
                className="py-2 px-3 rounded-xl bg-cyan-950 hover:bg-cyan-900 text-cyan-300 border border-cyan-800 font-semibold text-xs transition flex items-center justify-center gap-2"
              >
                {cameraLoading ? 'Switching Stream...' : 'Switch Camera Source'}
              </button>
            </form>
          </div>

          {/* Card 5: Real-Time Location Awareness (Ola Maps) */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
              <h3 className="font-bold text-sm text-slate-200 flex items-center gap-2">
                <span className="text-amber-400">📍</span> Live Location (Ola Maps)
              </h3>
              <button
                onClick={fetchLocation}
                disabled={locationLoading}
                className="text-[11px] text-amber-400 hover:text-amber-300 hover:underline flex items-center gap-1 font-medium"
              >
                {locationLoading ? 'Locating...' : 'Refresh'}
              </button>
            </div>

            {locationData && locationData.success ? (
              <div className="flex flex-col gap-2 bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                <div className="text-xs text-slate-200 font-medium leading-relaxed">
                  {locationData.address}
                </div>
                {locationData.coordinates && (
                  <div className="text-[10px] font-mono text-amber-400/90 flex gap-3">
                    <span>Lat: {locationData.coordinates.lat?.toFixed(4)}</span>
                    <span>Lng: {locationData.coordinates.lng?.toFixed(4)}</span>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-xs text-slate-500 py-2 italic">
                {locationLoading ? 'Pinpointing location via Ola Maps...' : 'Click Refresh to retrieve real-time location.'}
              </p>
            )}
          </div>

        </div>
      </main>
    </div>
  );
}
