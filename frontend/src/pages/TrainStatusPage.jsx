import { Link, useNavigate, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Alert, Button, Progress, Result, Space, Typography } from 'antd'
import { api } from '../api'

export default function TrainStatusPage() {
  const { jobId } = useParams()
  const navigate = useNavigate()

  const { data: status, error, isLoading } = useQuery({
    queryKey: ['train', jobId],
    queryFn: () => api.trainStatus(jobId),
    refetchInterval: (query) => {
      const s = query.state.data?.status
      if (s === 'succeeded' || s === 'failed') return false
      return 2000
    },
  })

  if (error) return <Alert type="error" message={error.message} showIcon />
  if (isLoading || !status) return <Progress percent={0} status="active" />

  if (status.status === 'succeeded') {
    return (
      <Result
        status="success"
        title={`${status.ticker} 训练完成`}
        subTitle={status.message}
        extra={[
          <Button
            type="primary"
            key="predict"
            onClick={() => navigate(`/result/${encodeURIComponent(status.ticker)}`)}
          >
            查看预测
          </Button>,
          <Button key="back">
            <Link to="/predict">返回工作台</Link>
          </Button>,
        ]}
      />
    )
  }

  if (status.status === 'failed') {
    return (
      <Result
        status="error"
        title="训练失败"
        subTitle={status.error || status.message}
        extra={
          <Button type="primary">
            <Link to="/predict">返回工作台</Link>
          </Button>
        }
      />
    )
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>
        训练中：{status.ticker}（#{status.job_id}）
      </Typography.Title>
      <Progress percent={status.progress || 0} status="active" />
      <Typography.Text>{status.message || status.status}</Typography.Text>
    </Space>
  )
}
