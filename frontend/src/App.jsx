import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { ConfigProvider, App as AntApp } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import AppLayout from './layout/AppLayout'
import HomePage from './pages/HomePage'
import PredictPage from './pages/PredictPage'
import TrainStatusPage from './pages/TrainStatusPage'
import ResultPage from './pages/ResultPage'
import TickersPage from './pages/TickersPage'
import './App.css'

export default function App() {
  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: '#1677ff',
          borderRadius: 8,
          fontFamily:
            '"IBM Plex Sans", "Noto Sans SC", system-ui, -apple-system, sans-serif',
        },
      }}
    >
      <AntApp>
        <BrowserRouter>
          <Routes>
            <Route element={<AppLayout />}>
              <Route path="/" element={<HomePage />} />
              <Route path="/predict" element={<PredictPage />} />
              <Route path="/train/:jobId" element={<TrainStatusPage />} />
              <Route path="/result/:ticker" element={<ResultPage />} />
              <Route path="/tickers" element={<TickersPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AntApp>
    </ConfigProvider>
  )
}
