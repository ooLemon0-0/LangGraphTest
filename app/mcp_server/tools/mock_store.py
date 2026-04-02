"""Centralized mock data store for MCP tools.

The store keeps list/detail responses consistent so multi-step LangGraph traces
stay debuggable and useful while the backend is still mocked.
"""

from __future__ import annotations

from copy import deepcopy


AGENT_ZHANGSAN = "agent_\u5f20\u4e09"
NAME_ZHANGSAN = "\u5f20\u4e09"

MOCK_AGENTS: dict[str, dict[str, str]] = {
    AGENT_ZHANGSAN: {"agent_id": AGENT_ZHANGSAN, "agent_name": NAME_ZHANGSAN},
    "agent_demo_001": {"agent_id": "agent_demo_001", "agent_name": "Demo Agent"},
}

MOCK_HOUSES_BY_AGENT: dict[str, list[dict[str, object]]] = {
    AGENT_ZHANGSAN: [
        {
            "house_id": f"{AGENT_ZHANGSAN}_house_001",
            "name": "Mock Maple House",
            "price": 680000.0,
            "currency": "USD",
            "agent_id": AGENT_ZHANGSAN,
            "status": "mock_active",
        },
        {
            "house_id": f"{AGENT_ZHANGSAN}_house_002",
            "name": "Mock River House",
            "price": 820000.0,
            "currency": "USD",
            "agent_id": AGENT_ZHANGSAN,
            "status": "mock_active",
        },
    ],
    "agent_demo_001": [
        {
            "house_id": "house_demo_001",
            "name": "Mock Harbor House",
            "price": 750000.0,
            "currency": "USD",
            "agent_id": "agent_demo_001",
            "status": "mock_active",
        }
    ],
}


def resolve_agent_by_name(agent_name: str) -> dict[str, str]:
    """Return a stable mock agent record and create one if needed."""
    agent_id = f"agent_{agent_name.strip().lower().replace(' ', '_')}"
    if agent_id not in MOCK_AGENTS:
        MOCK_AGENTS[agent_id] = {"agent_id": agent_id, "agent_name": agent_name}
        MOCK_HOUSES_BY_AGENT[agent_id] = [
            {
                "house_id": f"{agent_id}_house_001",
                "name": f"{agent_name} \u4e00\u53f7\u623f\u6e90",
                "price": 660000.0,
                "currency": "USD",
                "agent_id": agent_id,
                "status": "mock_active",
            },
            {
                "house_id": f"{agent_id}_house_002",
                "name": f"{agent_name} \u4e8c\u53f7\u623f\u6e90",
                "price": 790000.0,
                "currency": "USD",
                "agent_id": agent_id,
                "status": "mock_active",
            },
        ]
    return deepcopy(MOCK_AGENTS[agent_id])


def get_houses_for_agent(agent_id: str) -> list[dict[str, object]]:
    """Return a stable list of houses for one agent."""
    houses = MOCK_HOUSES_BY_AGENT.get(agent_id)
    if houses is None:
        houses = [
            {
                "house_id": f"{agent_id}_house_001",
                "name": "Mock Maple House",
                "price": 680000.0,
                "currency": "USD",
                "agent_id": agent_id,
                "status": "mock_active",
            },
            {
                "house_id": f"{agent_id}_house_002",
                "name": "Mock River House",
                "price": 820000.0,
                "currency": "USD",
                "agent_id": agent_id,
                "status": "mock_active",
            },
        ]
        MOCK_HOUSES_BY_AGENT[agent_id] = houses
    return deepcopy(houses)


def get_house_detail_record(house_id: str) -> dict[str, object]:
    """Return one stable house detail record by house id."""
    for houses in MOCK_HOUSES_BY_AGENT.values():
        for house in houses:
            if house["house_id"] == house_id:
                return deepcopy(house)

    inferred_agent_id = house_id.rsplit("_house_", 1)[0] if "_house_" in house_id else "agent_demo_001"
    generated = {
        "house_id": house_id,
        "name": "Mock Harbor House",
        "price": 750000.0,
        "currency": "USD",
        "agent_id": inferred_agent_id,
        "status": "mock_active",
    }
    MOCK_HOUSES_BY_AGENT.setdefault(inferred_agent_id, []).append(generated)
    return deepcopy(generated)


def update_house_name_record(house_id: str, new_name: str) -> dict[str, object]:
    """Update the in-memory house name for consistent later reads."""
    house = get_house_detail_record(house_id)
    house["name"] = new_name
    _persist_house(house)
    return house


def update_house_price_record(house_id: str, new_price: float, currency: str) -> dict[str, object]:
    """Update the in-memory house price for consistent later reads."""
    house = get_house_detail_record(house_id)
    house["price"] = new_price
    house["currency"] = currency
    _persist_house(house)
    return house


def _persist_house(updated_house: dict[str, object]) -> None:
    """Write one updated house back into the centralized mock store."""
    agent_id = str(updated_house["agent_id"])
    houses = MOCK_HOUSES_BY_AGENT.setdefault(agent_id, [])
    for index, house in enumerate(houses):
        if house["house_id"] == updated_house["house_id"]:
            houses[index] = deepcopy(updated_house)
            return
    houses.append(deepcopy(updated_house))
