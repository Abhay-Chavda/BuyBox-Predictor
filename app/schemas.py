from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    competitors_history: list[dict] = Field(alias="CompetitorsHistory")
    seller_id: int | str = Field(alias="SellerId")
    buybox_history_id: int | str = Field(alias="BuyboxHistoryId")
    min_price: float = Field(alias="MinPrice")
    max_price: float = Field(alias="MaxPrice")

    model_config = {"populate_by_name": True}
