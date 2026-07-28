import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Alert, Col, Row, Spin, Table, Typography } from 'antd'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api'

export default function HomePage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['home'],
    queryFn: api.home,
    refetchInterval: 60_000,
  })

  const recent = data?.recent_stocks || []
  const series = data?.series || []

  const chartData = useMemo(() => {
    const map = new Map()
    for (const s of series) {
      for (const p of s.points || []) {
        const row = map.get(p.date) || { date: p.date }
        row[s.name] = p.close
        map.set(p.date, row)
      }
    }
    return Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date))
  }, [series])

  const columns = [
    { title: '代码', dataIndex: 'Ticker', key: 'Ticker' },
    { title: '名称', dataIndex: 'Name', key: 'Name' },
    { title: '现价', dataIndex: 'Close', key: 'Close' },
    { title: '涨跌幅', dataIndex: 'Percent_Change', key: 'Percent_Change' },
    { title: '成交量', dataIndex: 'Volume', key: 'Volume' },
  ]

  if (isLoading) return <Spin />
  if (error) return <Alert type="error" message={error.message} showIcon />

  return (
    <div>
      <Typography.Title level={3}>市场概览</Typography.Title>
      <Typography.Paragraph type="secondary">
        中国热门股票近 30 日收盘价与实时行情（腾讯行情接口）
      </Typography.Paragraph>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={14}>
          <div style={{ width: '100%', height: 360 }}>
            <ResponsiveContainer>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" hide />
                <YAxis domain={['auto', 'auto']} />
                <Tooltip />
                <Legend />
                {series.map((s, idx) => (
                  <Line
                    key={s.code}
                    type="monotone"
                    dataKey={s.name}
                    stroke={['#1677ff', '#52c41a', '#fa8c16', '#eb2f96'][idx % 4]}
                    dot={false}
                    strokeWidth={2}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Col>
        <Col xs={24} lg={10}>
          <Table
            size="small"
            rowKey={(r) => r.Ticker || JSON.stringify(r)}
            columns={columns}
            dataSource={recent}
            pagination={false}
          />
        </Col>
      </Row>
    </div>
  )
}
