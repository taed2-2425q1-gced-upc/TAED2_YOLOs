import {PredictionStats} from "@/infraestructure/types"

interface PredictionStatsProps {
  stats: PredictionStats
}

interface Stat {
  statName: string
  statValue: number
}

export const PredictionStatsContainer: React.FC<PredictionStatsProps> = ({
  stats,
}) => {
  if (!stats) {
    return null
  }

  const statsList = Object.entries(stats).map(([key, value]) => {
    return {statName: key, statValue: value}
  })

  return (
    <div className="flex flex-col mt-4">
      <h2>Inference stats</h2>
      <div className="flex flex-wrap gap-3 mt-1">
        {statsList.map((stat) => (
          <PredictionStat key={stat.statName} {...stat} />
        ))}
      </div>
    </div>
  )
}

export const PredictionStat: React.FC<Stat> = ({statName, statValue}) => {
  return (
    <div className="flex items-center gap-2 bg-gray-900 rounded-lg py-2 px-4">
      <h3>{statName}:</h3>
      <p>{statValue}</p>
    </div>
  )
}
