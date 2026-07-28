import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AutoComplete,
  Button,
  Card,
  Empty,
  Popconfirm,
  Space,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import { DeleteOutlined, PlusOutlined, RocketOutlined } from '@ant-design/icons'
import { api } from '../api'

function HorizonCell({ horizon }) {
  if (!horizon) {
    return <Typography.Text type="secondary">加载中</Typography.Text>
  }
  if (horizon.status && horizon.status !== 'ok') {
    return <Tag>{horizon.status === 'missing' ? '未训练' : '不可用'}</Tag>
  }
  const up = horizon.direction === 'up'
  const pct =
    typeof horizon.probability === 'number'
      ? `${(horizon.probability * 100).toFixed(1)}%`
      : '-'
  return (
    <Space size={6}>
      <Tag color={up ? 'success' : 'error'} style={{ marginInlineEnd: 0 }}>
        {up ? '看涨' : '看跌'}
      </Tag>
      <Typography.Text>{pct}</Typography.Text>
    </Space>
  )
}

function pickHorizon(forecast, key) {
  return (forecast?.horizons || []).find((h) => h.key === key) || null
}

export default function PredictPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [options, setOptions] = useState([])
  const [ticker, setTicker] = useState('')
  const [deletingKey, setDeletingKey] = useState(null)

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    refetchInterval: 15_000,
  })
  const models = data?.items || []

  const forecastQueries = useQueries({
    queries: models.map((m) => ({
      queryKey: ['predict-brief', m.market, m.ticker],
      queryFn: () => api.predict(m.ticker),
      enabled: Boolean(m.ticker),
      staleTime: 60_000,
      refetchInterval: 60_000,
      retry: 1,
    })),
  })

  const forecastByKey = useMemo(() => {
    const map = {}
    models.forEach((m, i) => {
      const key = `${m.market}-${m.ticker}`
      map[key] = forecastQueries[i]?.data || null
      map[`${key}:loading`] = forecastQueries[i]?.isLoading || forecastQueries[i]?.isFetching
      map[`${key}:error`] = forecastQueries[i]?.error || null
    })
    return map
  }, [models, forecastQueries])

  const trainMutation = useMutation({
    mutationFn: (symbol) => api.train(symbol),
    onSuccess: (res) => {
      message.success(res.reused ? '已有进行中的训练任务' : '已加入训练队列')
      queryClient.invalidateQueries({ queryKey: ['models'] })
      navigate(`/train/${res.job_id}`)
    },
    onError: (e) => message.error(e.message),
  })

  const deleteMutation = useMutation({
    mutationFn: ({ symbol, market }) => api.deleteModel(symbol, market),
    onMutate: ({ symbol, market }) => {
      setDeletingKey(`${market}-${symbol}`)
    },
    onSuccess: (_data, vars) => {
      message.success('模型已删除')
      queryClient.invalidateQueries({ queryKey: ['models'] })
      queryClient.invalidateQueries({
        queryKey: ['predict-brief', vars.market, vars.symbol],
      })
    },
    onError: (e) => message.error(e.message),
    onSettled: () => setDeletingKey(null),
  })

  const onSearch = async (value) => {
    if (!value || value.length < 1) {
      setOptions([])
      return
    }
    try {
      const suggest = await api.suggest(value)
      setOptions(
        (suggest.items || []).map((item) => ({
          value: item.symbol,
          label: item.label || `${item.symbol} ${item.name || ''}`,
        }))
      )
    } catch {
      setOptions([])
    }
  }

  const startTrain = () => {
    if (!ticker.trim()) {
      message.warning('请输入股票代码')
      return
    }
    trainMutation.mutate(ticker.trim())
  }

  const goPredict = (symbol) => {
    navigate(`/result/${encodeURIComponent(symbol)}`)
  }

  const columns = [
    {
      title: '代码',
      dataIndex: 'ticker',
      fixed: 'left',
      width: 100,
      render: (t, row) => (
        <Button type="link" style={{ padding: 0 }} onClick={() => goPredict(t)}>
          {t}
          {row.name ? (
            <Typography.Text type="secondary" style={{ marginLeft: 6 }}>
              {row.name}
            </Typography.Text>
          ) : null}
        </Button>
      ),
    },
    {
      title: '市场',
      dataIndex: 'market_label',
      width: 88,
    },
    {
      title: '日 · 涨跌 / 概率',
      key: 'day',
      width: 150,
      render: (_v, row) => {
        const key = `${row.market}-${row.ticker}`
        if (forecastByKey[`${key}:error`]) {
          return <Typography.Text type="danger">失败</Typography.Text>
        }
        if (forecastByKey[`${key}:loading`] && !forecastByKey[key]) {
          return <Typography.Text type="secondary">…</Typography.Text>
        }
        return <HorizonCell horizon={pickHorizon(forecastByKey[key], 'day')} />
      },
    },
    {
      title: '周 · 涨跌 / 概率',
      key: 'week',
      width: 150,
      render: (_v, row) => {
        const key = `${row.market}-${row.ticker}`
        if (forecastByKey[`${key}:loading`] && !forecastByKey[key]) {
          return <Typography.Text type="secondary">…</Typography.Text>
        }
        return <HorizonCell horizon={pickHorizon(forecastByKey[key], 'week')} />
      },
    },
    {
      title: '月 · 涨跌 / 概率',
      key: 'month',
      width: 150,
      render: (_v, row) => {
        const key = `${row.market}-${row.ticker}`
        if (forecastByKey[`${key}:loading`] && !forecastByKey[key]) {
          return <Typography.Text type="secondary">…</Typography.Text>
        }
        return <HorizonCell horizon={pickHorizon(forecastByKey[key], 'month')} />
      },
    },
    {
      title: '验证准确率(日/周/月)',
      key: 'val_acc',
      width: 160,
      render: (_v, row) => {
        const fmt = (v) => (typeof v === 'number' ? `${v.toFixed(1)}%` : '-')
        return (
          <Typography.Text style={{ fontSize: 12 }}>
            {fmt(row.val_direction_acc)} / {fmt(row.val_week_direction_acc)} /{' '}
            {fmt(row.val_month_direction_acc)}
          </Typography.Text>
        )
      },
    },
    {
      title: '训练时间',
      dataIndex: 'trained_at',
      width: 180,
      ellipsis: true,
      render: (t) => t || '-',
    },
    {
      title: '操作',
      key: 'actions',
      fixed: 'right',
      width: 140,
      render: (_v, row) => (
        <Space size={0}>
          <Button type="link" onClick={() => goPredict(row.ticker)}>
            详情
          </Button>
          <Popconfirm
            title="确认删除该模型？"
            description="将删除磁盘产物与数据库记录，此操作不可恢复。"
            okText="删除"
            cancelText="取消"
            okButtonProps={{ danger: true }}
            onConfirm={() =>
              deleteMutation.mutate({ symbol: row.ticker, market: row.market })
            }
          >
            <Button
              type="link"
              danger
              icon={<DeleteOutlined />}
              loading={deletingKey === `${row.market}-${row.ticker}`}
            >
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <Typography.Title level={3}>训练与预测工作台</Typography.Title>
      <Card style={{ marginBottom: 24 }}>
        <Space wrap size="middle">
          <AutoComplete
            style={{ width: 280 }}
            options={options}
            onSearch={onSearch}
            onSelect={(v) => setTicker(v)}
            value={ticker}
            onChange={setTicker}
            placeholder="输入代码或名称，如 600519 / 茅台"
          />
          <Button
            type="primary"
            icon={<PlusOutlined />}
            loading={trainMutation.isPending}
            onClick={startTrain}
          >
            训练模型
          </Button>
          <Button
            icon={<RocketOutlined />}
            onClick={() => goPredict(ticker.trim())}
            disabled={!ticker.trim()}
          >
            直接预测
          </Button>
        </Space>
      </Card>

      <Typography.Title level={4}>已训练模型</Typography.Title>
      {models.length === 0 && !isLoading ? (
        <Empty description="暂无已训练模型，请先训练" />
      ) : (
        <Table
          rowKey={(r) => `${r.market}-${r.ticker}`}
          loading={isLoading || isFetching}
          columns={columns}
          dataSource={models}
          pagination={false}
          scroll={{ x: 1100 }}
          size="middle"
        />
      )}
    </div>
  )
}
