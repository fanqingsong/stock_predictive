import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Alert, Input, Spin, Table, Typography } from 'antd'
import { api } from '../api'

export default function TickersPage() {
  const [q, setQ] = useState('')
  const { data, isLoading, error } = useQuery({
    queryKey: ['tickers'],
    queryFn: api.tickers,
    staleTime: 5 * 60_000,
  })
  const items = data?.items || []

  const filtered = useMemo(() => {
    const keyword = q.trim().toLowerCase()
    if (!keyword) return items.slice(0, 200)
    return items
      .filter(
        (r) =>
          String(r.Symbol || '').toLowerCase().includes(keyword) ||
          String(r.Name || '').toLowerCase().includes(keyword)
      )
      .slice(0, 200)
  }, [items, q])

  const columns = [
    { title: 'Symbol', dataIndex: 'Symbol', key: 'Symbol' },
    { title: 'Name', dataIndex: 'Name', key: 'Name' },
    { title: 'Market', dataIndex: 'Market', key: 'Market' },
  ]

  if (isLoading) return <Spin />
  if (error) return <Alert type="error" message={error.message} showIcon />

  return (
    <div>
      <Typography.Title level={3}>Ticker 列表</Typography.Title>
      <Input.Search
        placeholder="搜索代码或名称"
        allowClear
        style={{ maxWidth: 360, marginBottom: 16 }}
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />
      <Table
        rowKey={(r) => `${r.Market}-${r.Symbol}`}
        columns={columns}
        dataSource={filtered}
        pagination={{ pageSize: 20 }}
      />
    </div>
  )
}
